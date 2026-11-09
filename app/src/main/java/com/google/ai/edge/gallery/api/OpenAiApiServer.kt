/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.gallery.api

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import android.util.Log
import com.google.ai.edge.gallery.agent.AgentEvent
import com.google.ai.edge.gallery.agent.AgentExecutionContext
import com.google.ai.edge.gallery.agent.AgentRequest
import com.google.ai.edge.gallery.agent.AgentRuntimeExecutor
import com.google.ai.edge.gallery.agent.AiChatExecutor
import com.google.ai.edge.gallery.agent.Attachment
import com.google.ai.edge.gallery.data.BuiltInTaskId
import com.google.ai.edge.gallery.data.ConfigKeys
import com.google.ai.edge.gallery.data.DEFAULT_MAX_TOKEN
import com.google.ai.edge.gallery.data.KeepAliveStore
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.RuntimeType
import com.google.ai.edge.gallery.keepalive.ApiServerService
import com.google.ai.edge.gallery.runtime.SamplerOverride
import com.google.ai.edge.gallery.runtime.runtimeHelper
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.ToolCall
import com.google.ai.edge.litertlm.ToolProvider
import com.google.ai.edge.litertlm.tool
import dagger.hilt.android.qualifiers.ApplicationContext
import fi.iki.elonen.NanoHTTPD
import java.net.Inet4Address
import java.net.NetworkInterface
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.locks.ReentrantLock
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

private const val TAG = "AGOpenAiApiServer"
private const val DEFAULT_PORT = 8080

/**
 * An OpenAI-compatible HTTP server (backed by NanoHTTPD) that exposes the on-device LLM loaded by
 * the Gallery app over the local network, so that any OpenAI-compatible client (curl, the Python
 * `openai` library, etc.) can call it via [/v1/chat/completions].
 *
 * The server does NOT initialize a model itself. It reuses the model session already initialized
 * in the app UI (the user must first download and initialize a model in the chat screen). Only one
 * model is served at a time for chat completions — the one set via [setActiveModel].
 *
 * The list of models advertised by `/v1/models` (and `/v1/chat/completions/models`) is just the
 * single model currently being served (see [setActiveModel]); no other downloaded models are
 * advertised, so clients only ever select a model that is actually running.
 *
 * Supports both streaming (SSE) and non-streaming OpenAI chat completions.
 */
@Singleton
open class OpenAiApiServer
@Inject
constructor(
  @ApplicationContext private val context: Context,
  @AiChatExecutor private val executor: AgentRuntimeExecutor,
) : NanoHTTPD(DEFAULT_PORT) {
  @Volatile private var activeModel: Model? = null
  /** Serializes inference: only one request runs at a time; others block and wait in queue. */
  private val inferenceLock = ReentrantLock()

  /**
   * Runs the critical section off the HTTP thread so a streaming response can be returned while
   * tokens are still being generated. The single thread plus [inferenceLock] keeps the existing
   * serial queue: a second request waits until the first generation (and its stream) finishes.
   */
  private val inferenceExecutor =
    Executors.newSingleThreadExecutor { runnable ->
      Thread(runnable, "openai-inference").apply { isDaemon = true }
    }

  /** When non-null, requests to /v1 endpoints must present this token via Authorization Bearer. */
  @Volatile private var authToken: String? = null

  /** Diagnostics of the most recent `/v1/chat/completions` request, exposed via `/debug` so I can
   *  troubleshoot tool calling on-device without needing logcat. */
  data class RequestDiagnostics(
    var hasToolsField: Boolean = false,
    var toolCount: Int = 0,
    var toolNames: List<String> = emptyList(),
    var toolDescriptionJson: String = "",
    var automaticToolCalling: Boolean = true,
    var toolCallEvents: Int = 0,
    var streamTokenEvents: Int = 0,
    var error: String? = null,
  )

  @Volatile private var lastDiagnostics = RequestDiagnostics()

  val isRunning: Boolean
    get() = isAlive()

  /**
   * Observable running state, kept in sync with the actual HTTP server. Screens observe this (via
   * [com.google.ai.edge.gallery.api.ApiServerViewModel]) so any ViewModel instance reflects the
   * live server state, e.g. to show the "back to API" affordance on the home page while the server
   * is running.
   */
  private val _runState = MutableStateFlow(isAlive())
  val runState: StateFlow<Boolean> = _runState.asStateFlow()

  /** Enables/disables bearer-token auth for the /v1 endpoints. */
  fun setAuthToken(token: String?) {
    authToken = token?.takeIf { it.isNotBlank() }
    Log.d(TAG, "Auth token set: " + if (authToken != null) "enabled" else "disabled")
  }

  /** Sets the model (with its task) that the HTTP server should serve. */
  @Suppress("UNUSED_PARAMETER")
  fun setActiveModel(model: Model, taskId: String = BuiltInTaskId.LLM_CHAT) {
    activeModel = model
    Log.d(TAG, "Active model set to ${model.name} (task=$taskId)")
  }

  /** The name of the model currently being served, or null if none was set. */
  fun getActiveModelName(): String? = activeModel?.name

  /**
   * True while the HTTP server is running and serving [modelName].
   *
   * Model instances are owned by the UI task scaffold, which normally cleans a model up when the
   * user navigates away or switches models. When a model is actively served by the API server it
   * must stay resident (its `instance` must not be torn down) so the server keeps answering calls
   * even after the user leaves the API page. Callers that clean up models should consult this
   * before tearing a model down.
   */
  fun isServingModel(modelName: String): Boolean =
    isAlive() && activeModel?.name == modelName

  /**
   * The model name exposed to API clients, without the import file extension.
   *
   * Imported models use the source file name (e.g. `foo.litertlm`) as their internal [Model.name].
   * API responses (`/v1/models` id, chat `model` field) present the cleaned-up display name instead
   * so clients never see the `.litertlm` suffix.
   */
  private fun apiModelName(model: Model): String = model.displayName.ifEmpty { model.name }

  /** The base URL if the server is running, or null otherwise. */
  fun currentBaseUrl(): String? = if (isAlive()) baseUrl() else null

  /**
   * Starts the HTTP server on the configured port.
   *
   * @return the base URL (e.g. "http://192.168.1.5:8080") if started successfully, or null if the
   *   server could not be started.
   */
   fun startServer(port: Int = DEFAULT_PORT): String? {
    if (isAlive()) {
      Log.w(TAG, "Server already running")
      return baseUrl()
    }
    try {
      super.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false)
    } catch (e: Exception) {
      Log.e(TAG, "Failed to start server", e)
      return null
    }
    Log.d(TAG, "OpenAI-compatible server started on port $port")
    _runState.value = true
    // Keep the process alive: run the server inside a foreground service with a persistent
    // notification so it survives the screen being off or the app being backgrounded. Enable
    // independent keep-alive so the app stays resident even after the server stops (until the
    // process is killed), keeping the local API reachable without a cold start. This is
    // best-effort and independent of the server's own start result.
    try {
      KeepAliveStore.setEnabled(context, true)
      ApiServerService.start(context, getActiveModelName())
    } catch (e: Exception) {
      Log.w(TAG, "Failed to start keep-alive service", e)
    }
    return baseUrl()
  }

  /** Stops the HTTP server. Safe to call even if it is not running. */
  fun stopServer() {
    // Independent keep-alive stays on after the server stops (until the process is killed), so the
    // foreground service keeps the app resident and the local API reachable without a cold start.
    // We therefore do NOT tear down the keep-alive service here.
    if (isAlive()) {
      try {
        super.stop()
      } catch (e: Exception) {
        Log.w(TAG, "Error stopping server", e)
      }
      Log.d(TAG, "Server stopped")
    }
    // Reflect the server state so observing ViewModels update their running/URL state. Emitted
    // regardless of whether it was actually alive above.
    _runState.value = false
    // Keep-alive stays on (until the process is killed), so re-trigger the foreground service so
    // it rebuilds its notification: with the server no longer running it switches to the
    // keep-alive-only message instead of the stale "serving" message. This also runs when the
    // server died unexpectedly (isAlive() is false), so the notification never keeps showing a
    // stale "serving" state after the server is gone.
    ApiServerService.start(context, null)
  }

  /** Best-effort base URL of the running server (first private IPv4 found). */
  private fun baseUrl(): String? {
    val port = getListeningPort()
    val ip = getLocalIpV4() ?: return null
    return "http://$ip:$port"
  }

  private fun getLocalIpV4(): String? {
    try {
      NetworkInterface.getNetworkInterfaces().toList().forEach { nif ->
        if (!nif.isLoopback && nif.isUp) {
          val addr = nif.inetAddresses.toList().firstOrNull { it is Inet4Address }
          if (addr != null) return addr.hostAddress
        }
      }
    } catch (e: Exception) {
      Log.w(TAG, "Failed to enumerate network interfaces", e)
    }
    return null
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // NanoHTTPD routing
  ////////////////////////////////////////////////////////////////////////////////////////////////

  override fun serve(
    session: IHTTPSession,
  ): Response {
    val uri = session.uri
    val method = session.method
    // Handle CORS preflight (browser sends OPTIONS before a cross-origin POST with
    // custom headers like Authorization). Must be answered BEFORE auth so the browser
    // isn't blocked; the actual request is still authenticated.
    if (method == Method.OPTIONS) {
      return corsResponse()
    }
    // Protect /v1/* endpoints with optional bearer token auth.
    if (uri.startsWith("/v1") && !isAuthorized(session)) {
      return jsonError(Response.Status.UNAUTHORIZED, "Unauthorized")
    }
    return when {
      uri == "/health" -> newFixedLengthResponse(Response.Status.OK, "text/plain", "ok")
      uri == "/debug" -> handleDebug()
      uri == "/v1/models" || uri == "/v1/chat/completions/models" -> handleModels()
      uri == "/v1/chat/completions" && method == Method.POST ->
        handleChatCompletions(session)
      uri.startsWith("/v1") ->
        jsonError(Response.Status.NOT_FOUND, "Not found")
      else -> newFixedLengthResponse(
        Response.Status.OK,
        "text/plain",
        listOf("/v1/chat/completions", "/v1/models", "/v1/chat/completions/models", "/health")
          .joinToString("\n"),
      )
    }
  }

  /** Empty 200 response with CORS headers, used to answer browser OPTIONS preflight. */
  private fun corsResponse(): Response {
    val res = newFixedLengthResponse(Response.Status.OK, "text/plain", "")
    res.addHeader("Access-Control-Allow-Origin", "*")
    res.addHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    res.addHeader("Access-Control-Allow-Headers", "Authorization, Content-Type")
    res.addHeader("Access-Control-Max-Age", "86400")
    return res
  }

  /** Returns true if no token is set, or the request carries a matching Authorization Bearer token. */
  private fun isAuthorized(session: IHTTPSession): Boolean {
    val expected = authToken ?: return true
    val header = session.headers["authorization"] ?: session.headers["Authorization"] ?: return false
    val provided = header.removePrefix("Bearer ").removePrefix("bearer ").trim()
    return provided.isNotEmpty() && provided == expected
  }

  /**
   * Lists the model currently being served by this server (set via [setActiveModel]). Only that one
   * model is advertised, because the server serves a single model at a time.
   */
  private fun handleModels(): Response {
    // Advertise only the model currently being served (set via setActiveModel). The server serves
    // exactly one model at a time, so listing every downloaded model here would mislead clients
    // into selecting a model that is not actually running.
    val active = getActiveOrNull()
    val items = if (active != null) modelEntryJson(active) else ""
    val body = "{\"object\":\"list\",\"data\":[$items]}"
    return jsonResponse(Response.Status.OK, body)
  }

  /** Serializes a model into an OpenAI `model` list entry. */
  private fun modelEntryJson(model: Model): String {
    return "{\"id\":\"${escapeJson(apiModelName(model))}\",\"object\":\"model\"," +
      "\"created\":${System.currentTimeMillis() / 1000},\"owned_by\":\"local\"}"
  }

  /** Returns diagnostics about the last chat completion request (for tool-calling debugging). */
  private fun handleDebug(): Response {
    val d = lastDiagnostics
    val body =
      "{\"hasToolsField\":${d.hasToolsField},\"toolCount\":${d.toolCount}," +
        "\"toolNames\":${d.toolNames.joinToString(",") { "\"${escapeJson(it)}\"" }.let { "[$it]" }}," +
        "\"automaticToolCalling\":${d.automaticToolCalling}," +
        "\"toolCallEvents\":${d.toolCallEvents},\"streamTokenEvents\":${d.streamTokenEvents}," +
        "\"toolDescriptionJson\":${d.toolDescriptionJson.let { "\"${escapeJson(it)}\"" }}," +
        "\"error\":${d.error?.let { "\"${escapeJson(it)}\"" } ?: "null"}}"
    return jsonResponse(Response.Status.OK, body)
  }

  private fun handleChatCompletions(session: IHTTPSession): Response {
    val model = getActiveOrNull()
    if (model == null) {
      return jsonError(
        Response.Status.BAD_REQUEST,
        "No active model. Please open the chat screen and initialize a model first.",
      )
    }

    val body = readRequestBody(session)
    val parsed =
      when (val outcome = parseChatBody(body)) {
        is ParseOutcome.Rejected -> {
          lastDiagnostics = RequestDiagnostics(error = outcome.message)
          return jsonError(Response.Status.BAD_REQUEST, outcome.message)
        }
        is ParseOutcome.Ok -> outcome.request
      }
    val advertised = apiModelName(model)
    val modelError = validateRequestedModel(parsed.requestedModel, advertised)
    if (modelError != null) {
      return jsonError(Response.Status.NOT_FOUND, modelError, type = "model_not_found")
    }

    val images = mutableListOf<Bitmap>()
    for (url in parsed.imageDataUrls) {
      val bitmap = decodeImageFromDataUrl(url)
      if (bitmap == null) {
        return jsonError(Response.Status.BAD_REQUEST, "Could not decode image_url")
      }
      images.add(bitmap)
    }
    val samplerOverride =
      if (
        parsed.temperature != null ||
          parsed.topP != null ||
          parsed.topK != null ||
          parsed.seed != null
      ) {
        SamplerOverride(
          temperature = parsed.temperature,
          topK = parsed.topK,
          topP = parsed.topP,
          seed = parsed.seed,
        )
      } else {
        null
      }
    if (samplerOverride != null && model.runtimeType == RuntimeType.AICORE) {
      return jsonError(
        Response.Status.BAD_REQUEST,
        "Per-request temperature, top_p, top_k, and seed are not supported for AICore models.",
      )
    }
    if (
      model.runtimeType == RuntimeType.AICORE &&
        (parsed.repetitionPenalty != null ||
          parsed.presencePenalty != null ||
          parsed.frequencyPenalty != null)
    ) {
      return jsonError(
        Response.Status.BAD_REQUEST,
        "repetition_penalty, presence_penalty, and frequency_penalty are not supported for AICore models.",
      )
    }
    val tools = parseTools(parsed.source)
    val request =
      ChatRequest(
        prompt = parsed.prompt,
        historyMessages = buildHistory(parsed),
        images = images,
        stream = parsed.stream,
        samplerOverride = samplerOverride,
        maxOutputTokens = parsed.maxOutputTokens,
        repetitionPenalty = parsed.repetitionPenalty,
        presencePenalty = parsed.presencePenalty,
        frequencyPenalty = parsed.frequencyPenalty,
        hadModelInput = parsed.hadModelInput,
        tools = tools,
        hasToolsField = (parsed.source["tools"] as? JsonArray)?.isNotEmpty() == true,
        toolNames = extractToolNames(parsed.source),
        toolDescriptionJson = extractToolDescriptionJson(parsed.source),
      )

    lastDiagnostics =
      RequestDiagnostics(
        hasToolsField = request.hasToolsField,
        toolCount = request.tools.size,
        toolNames = request.toolNames,
        toolDescriptionJson = request.toolDescriptionJson,
        automaticToolCalling = request.tools.isEmpty(),
      )
    Log.d(
      TAG,
      "Chat request: tools=${request.tools.size} names=${request.toolNames} " +
        "autoCall=${request.tools.isEmpty()} stream=${request.stream} " +
        "maxOutputTokens=${request.maxOutputTokens} temperature=${request.samplerOverride?.temperature}",
    )

    // Streaming must return the HTTP response before generation finishes, so the work is handed
    // to the inference thread. Non-streaming waits for that same thread, which is what keeps
    // requests ordered.
    return if (request.stream) {
      streamResponse(model, request)
    } else {
      try {
        inferenceExecutor.submit(Callable { runExclusive(model, request) }).get()
      } catch (e: Exception) {
        Log.e(TAG, "Chat completion failed", e)
        jsonError(Response.Status.INTERNAL_ERROR, e.message ?: "Inference failed", type = "server_error")
      }
    }
  }

  /** Rebuilds the model's conversation from [request.historyMessages] before inference. */
  private fun resetConversationForRequest(model: Model, request: ChatRequest) {
    val hasTools = request.tools.isNotEmpty()
    model.runtimeHelper.resetConversation(
      model = model,
      supportImage = request.images.isNotEmpty(),
      supportAudio = false,
      // System turns stay in [request.historyMessages] so their order relative to user and
      // assistant turns is preserved. ConversationConfig.systemInstruction would always prepend
      // them and is intentionally not used here.
      systemInstruction = null,
      tools = request.tools,
      automaticToolCalling = !hasTools,
      enableConversationConstrainedDecoding = hasTools,
      initialMessages = request.historyMessages,
      samplerOverride = request.samplerOverride,
    )
  }

  private fun buildHistory(parsed: ParsedChatRequest): List<Message> {
    val toolCallIdToName = mutableMapOf<String, String>()
    val messages = mutableListOf<Message>()
    for (msg in parsed.history) {
      when (msg.role) {
        "user" -> messages += Message.user(msg.text)
        "system" -> messages += Message.system(msg.text)
        "assistant" -> {
          if (msg.toolCalls.isNotEmpty()) {
            val calls = mutableListOf<ToolCall>()
            for (call in msg.toolCalls) {
              val args =
                runCatching {
                  @Suppress("UNCHECKED_CAST")
                  jsonElementToAny(Json.parseToJsonElement(call.arguments)) as Map<String, Any>
                }
                  .getOrDefault(emptyMap())
              calls += ToolCall(call.name, args)
              if (call.id.isNotEmpty()) toolCallIdToName[call.id] = call.name
            }
            messages += Message.model(Contents.of(Content.Text(msg.text)), calls, emptyMap())
          } else {
            messages += Message.model(msg.text)
          }
        }
        "tool" -> {
          val name = toolCallIdToName[msg.toolCallId] ?: msg.name.orEmpty()
          if (name.isNotEmpty()) {
            val response: Any =
              when (val content = msg.rawContent) {
                is JsonNull, null -> msg.text
                is JsonPrimitive -> content.content
                else -> runCatching { jsonElementToAny(content) }.getOrDefault(msg.text)
              }
            messages += Message.tool(Contents.of(Content.ToolResponse(name, response)))
          }
        }
      }
    }
    return messages
  }

  private fun runExclusive(model: Model, request: ChatRequest): Response {
    inferenceLock.lock()
    return try {
      try {
        resetConversationForRequest(model, request)
      } catch (e: IllegalArgumentException) {
        return jsonError(Response.Status.BAD_REQUEST, e.message ?: "Invalid request")
      } catch (e: Exception) {
        Log.w(TAG, "Failed to reset conversation for API request", e)
        return jsonError(
          Response.Status.INTERNAL_ERROR,
          "Failed to rebuild conversation: ${e.message ?: "unknown error"}",
          type = "server_error",
        )
      }
      runBlockingResponse(model, request)
    } finally {
      inferenceLock.unlock()
    }
  }

  /** Reads the request body from a NanoHTTPD session (POST JSON payload), decoded as UTF-8. */
  private fun readRequestBody(session: IHTTPSession): String {
    return try {
      // NanoHTTPD keeps the connection alive, so InputStream.readBytes() would block waiting
      // for EOF that never comes. Read exactly Content-Length bytes instead.
      val contentLength = session.headers["content-length"]?.toLongOrNull() ?: 0L
      val input = session.inputStream
      val bytes =
        if (contentLength > 0) {
          val buffer = ByteArray(contentLength.toInt())
          var total = 0
          while (total < buffer.size) {
            val n = input.read(buffer, total, buffer.size - total)
            if (n < 0) break
            total += n
          }
          buffer.copyOf(total)
        } else {
          // Fallback: read whatever is available (no Content-Length header).
          input.readBytes()
        }
      String(bytes, Charsets.UTF_8)
    } catch (e: Exception) {
      Log.w(TAG, "Failed to read request body", e)
      ""
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // Request parsing
  ////////////////////////////////////////////////////////////////////////////////////////////////

  private data class ChatRequest(
    val prompt: String,
    val historyMessages: List<Message>,
    val images: List<Bitmap>,
    val stream: Boolean,
    val samplerOverride: SamplerOverride?,
    val maxOutputTokens: Int?,
    val repetitionPenalty: Float?,
    val presencePenalty: Float?,
    val frequencyPenalty: Float?,
    val hadModelInput: Boolean,
    val tools: List<ToolProvider>,
    val hasToolsField: Boolean = false,
    val toolNames: List<String> = emptyList(),
    val toolDescriptionJson: String = "",
  )

  /**
   * Bridges an OpenAI `function` tool definition into LiteRT-LM's [OpenApiTool] so the model can
   * call it. Used in manual (client-driven) tool calling mode: the model emits a tool call, which
   * the server returns to the client; this [execute] is never invoked locally.
   */
  private class OpenAiFunctionTool(private val descriptionJson: String) : OpenApiTool {
    override fun getToolDescriptionJsonString(): String = descriptionJson
    override fun execute(json: String): String = ""
  }

  /**
   * Parses the OpenAI `tools` array into LiteRT-LM [ToolProvider]s. Each `function` definition is
   * bridged as an [OpenAiFunctionTool] whose description JSON mirrors the [ReflectionTool] schema
   * (`{name, description, parameters}`), so the model can call it in manual (client-driven) mode.
   */
  private fun parseTools(obj: JsonObject): List<ToolProvider> {
    val toolsEl = obj["tools"] ?: return emptyList()
    if (toolsEl !is JsonArray) return emptyList()
    val providers = mutableListOf<ToolProvider>()
    for (t in toolsEl) {
      val tObj = t.jsonObject
      if (tObj["type"]?.jsonPrimitive?.takeIf { it.isString }?.content != "function") continue
      val fn = tObj["function"]?.jsonObject ?: continue
      val name = fn["name"]?.jsonPrimitive?.takeIf { it.isString }?.content ?: continue
      val description = fn["description"]?.jsonPrimitive?.takeIf { it.isString }?.content ?: ""
      val parameters = fn["parameters"]?.toString() ?: "{}"
      val descriptionJson =
        "{\"name\":${jsonString(name)},\"description\":${jsonString(description)}," +
          "\"parameters\":$parameters}"
      providers.add(tool(OpenAiFunctionTool(descriptionJson)))
      Log.d(TAG, "Tool parsed: name=$name descriptionJson=$descriptionJson")
    }
    Log.d(TAG, "parseTools parsed ${providers.size} tool(s)")
    return providers
  }

  /** Extracts the tool function names from the request body (for diagnostics). */
  private fun extractToolNames(obj: JsonObject): List<String> {
    val toolsEl = obj["tools"] as? JsonArray ?: return emptyList()
    return toolsEl.mapNotNull { t ->
      val tObj = t as? JsonObject ?: return@mapNotNull null
      if (tObj["type"]?.jsonPrimitive?.takeIf { it.isString }?.content != "function") return@mapNotNull null
      tObj["function"]?.jsonObject?.get("name")?.jsonPrimitive?.takeIf { it.isString }?.content
    }
  }

  /** Extracts the first tool's full description JSON (for diagnostics). */
  private fun extractToolDescriptionJson(obj: JsonObject): String {
    val toolsEl = obj["tools"] as? JsonArray ?: return ""
    val first =
      toolsEl.firstNotNullOfOrNull { t ->
        val tObj = t as? JsonObject ?: return@firstNotNullOfOrNull null
        if (tObj["type"]?.jsonPrimitive?.takeIf { it.isString }?.content != "function") return@firstNotNullOfOrNull null
        tObj["function"]?.jsonObject
      } ?: return ""
    return buildString {
      append("{\"name\":${jsonString(first["name"]?.jsonPrimitive?.takeIf { it.isString }?.content ?: "")},")
      append("\"description\":${jsonString(first["description"]?.jsonPrimitive?.takeIf { it.isString }?.content ?: "")},")
      append("\"parameters\":${first["parameters"]?.toString() ?: "{}"}}")
    }
  }

  /** Converts a [JsonElement] into a plain [Any]: maps / lists / primitives / null. */
  private fun jsonElementToAny(element: JsonElement): Any {
    return when (element) {
      is JsonObject -> {
        val map = LinkedHashMap<String, Any>()
        for ((k, v) in element) map[k] = jsonElementToAny(v)
        map
      }
      is JsonArray -> element.map { jsonElementToAny(it) }
      is JsonPrimitive ->
        when {
          element.isString -> element.content
          element.booleanOrNull != null -> element.booleanOrNull!!
          element.intOrNull != null -> element.intOrNull!!
          element.doubleOrNull != null -> element.doubleOrNull!!
          else -> element.content
        }
      else -> ""
    }
  }

  /** Serializes a string into a JSON string literal (with escaping). */
  private fun jsonString(s: String): String = "\"${escapeJson(s)}\""

  /** Decodes a data:image/...;base64,... URL into a [Bitmap], or null if not decodable. */
  private fun decodeImageFromDataUrl(url: String): Bitmap? {
    return try {
      val comma = url.indexOf(',')
      val base64Data = if (comma >= 0) url.substring(comma + 1) else url
      val bytes = Base64.decode(base64Data, Base64.DEFAULT)
      BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    } catch (e: Exception) {
      Log.w(TAG, "Failed to decode image from data URL", e)
      null
    }
  }

  private fun getActiveOrNull(): Model? {
    val m = activeModel
    return if (m != null && m.instance != null) m else null
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // Generation
  ////////////////////////////////////////////////////////////////////////////////////////////////

  private class TurnState {
    val text = StringBuilder()
    val toolCalls = mutableListOf<ToolCall>()
    var errorMessage: String? = null
    var contextLimitError: Boolean = false
    var promptTooLong: Boolean = false
    var clientGone: Boolean = false
  }

  private fun streamResponse(model: Model, request: ChatRequest): Response {
    val body = SseBody(capacity = 4)
    inferenceExecutor.execute {
      inferenceLock.lock()
      try {
        try {
          resetConversationForRequest(model, request)
        } catch (e: IllegalArgumentException) {
          writeSse(body, sseData(errorJson(e.message ?: "Invalid request")))
          writeSse(body, sseDone())
          return@execute
        } catch (e: Exception) {
          Log.w(TAG, "Failed to reset conversation for API request", e)
          val message = "Failed to rebuild conversation: ${e.message ?: "unknown error"}"
          writeSse(body, sseData(errorJson(message, type = "server_error")))
          writeSse(body, sseDone())
          return@execute
        }
        val state = TurnState()
        try {
          runBlocking { collectEvents(model, request, state, sink = body) }
        } catch (e: kotlinx.coroutines.CancellationException) {
          if (!state.clientGone) Log.i(TAG, "Generation cancelled", e)
        } catch (e: Exception) {
          Log.e(TAG, "Streaming error", e)
          if (state.errorMessage == null) state.errorMessage = e.message ?: "unknown"
        }
        if (!state.clientGone) writeTerminal(body, model, request, state)
      } finally {
        body.complete()
        inferenceLock.unlock()
      }
    }
    val res = newChunkedResponse(Response.Status.OK, "text/event-stream", body)
    res.addHeader("Access-Control-Allow-Origin", "*")
    res.addHeader("Cache-Control", "no-cache")
    res.addHeader("Connection", "keep-alive")
    if (request.maxOutputTokens != null) {
      res.addHeader(OUTPUT_LIMIT_HEADER, OUTPUT_LIMIT_HEADER_VALUE)
    }
    return res
  }

  private fun runBlockingResponse(model: Model, request: ChatRequest): Response {
    val state = TurnState()
    try {
      runBlocking { collectEvents(model, request, state, sink = null) }
    } catch (e: Exception) {
      if (state.errorMessage == null) state.errorMessage = e.message ?: "unknown"
    }
    return completionResponse(model, request, state)
  }

  private suspend fun collectEvents(
    model: Model,
    request: ChatRequest,
    state: TurnState,
    sink: SseBody?,
  ) {
    val attachments = request.images.map { Attachment.ImageBitmap(it) }
    val metadata = mutableMapOf<String, Any>()
    if (request.tools.isNotEmpty()) metadata[AgentRequest.CAPTURE_TOOL_CALLS] = "true"
    request.maxOutputTokens?.let { metadata[AgentRequest.MAX_OUTPUT_TOKENS] = it }
    request.repetitionPenalty?.let { metadata[AgentRequest.REPETITION_PENALTY] = it }
    request.presencePenalty?.let { metadata[AgentRequest.PRESENCE_PENALTY] = it }
    request.frequencyPenalty?.let { metadata[AgentRequest.FREQUENCY_PENALTY] = it }
    val agentRequest =
      AgentRequest(query = request.prompt, attachments = attachments, metadata = metadata)
    val modelName = apiModelName(model)
    executor.executeStream(AgentExecutionContext(), agentRequest).collect { event ->
      if (state.clientGone) return@collect
      when (event) {
        is AgentEvent.StreamToken -> {
          lastDiagnostics.streamTokenEvents++
          if (event.token.isEmpty()) return@collect
          state.text.append(event.token)
          val chunk =
            sseData(completionChunk(modelName, textDeltaJson(event.token), null, null, false))
          if (sink != null && !writeSse(sink, chunk)) {
            state.clientGone = true
            requestStop(model)
            throw kotlinx.coroutines.CancellationException("client disconnected")
          }
        }
        is AgentEvent.ToolCalls -> {
          lastDiagnostics.toolCallEvents++
          val start = state.toolCalls.size
          state.toolCalls.addAll(event.toolCalls)
          val toolChunk = sseData(toolCallChunkJson(modelName, event.toolCalls, start))
          if (sink != null && !writeSse(sink, toolChunk)) {
            state.clientGone = true
            requestStop(model)
            throw kotlinx.coroutines.CancellationException("client disconnected")
          }
        }
        is AgentEvent.Error -> {
          if (state.clientGone) return@collect
          if (isPromptTooLongMessage(event.errorMessage)) {
            state.promptTooLong = true
            state.errorMessage = event.errorMessage
          } else if (isContextLimitMessage(event.errorMessage)) {
            state.contextLimitError = true
          } else {
            state.errorMessage = event.errorMessage
          }
        }
        else -> {}
      }
    }
  }

  private fun writeTerminal(body: SseBody, model: Model, request: ChatRequest, state: TurnState) {
    if (state.promptTooLong) {
      writeSse(
        body,
        sseData(errorJson(state.errorMessage ?: "Prompt exceeds the context limit")),
      )
      writeSse(body, sseDone())
      return
    }
    if (
      state.errorMessage != null &&
        !state.contextLimitError &&
        state.toolCalls.isEmpty()
    ) {
      writeSse(body, sseData(errorJson(state.errorMessage!!, type = "server_error")))
      writeSse(body, sseDone())
      return
    }
    val usage = measuredUsage(model, request, state)
    val finish = measuredFinish(model, request, state, usage)
    logFinish(finish, usage, contextLimit(model))
    val chunk =
      sseData(completionChunk(apiModelName(model), "{}", finish, usage, includeUsage = true))
    writeSse(body, chunk)
    writeSse(body, sseDone())
  }

  private fun completionResponse(model: Model, request: ChatRequest, state: TurnState): Response {
    if (state.promptTooLong) {
      return jsonError(
        Response.Status.BAD_REQUEST,
        state.errorMessage ?: "Prompt exceeds the context limit",
      )
    }
    if (
      state.errorMessage != null &&
        !state.contextLimitError &&
        state.toolCalls.isEmpty()
    ) {
      return jsonError(Response.Status.INTERNAL_ERROR, state.errorMessage!!, type = "server_error")
    }
    val usage = measuredUsage(model, request, state)
    val finish = measuredFinish(model, request, state, usage)
    logFinish(finish, usage, contextLimit(model))
    val messageJson =
      if (state.toolCalls.isNotEmpty()) {
        "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[${toolCallObjects(state.toolCalls)}]}"
      } else {
        "{\"role\":\"assistant\",\"content\":\"${escapeJson(state.text.toString())}\"}"
      }
    val response =
      jsonResponse(
        Response.Status.OK,
        completionJson(apiModelName(model), messageJson, finish, usage),
      )
    if (request.maxOutputTokens != null) {
      response.addHeader(OUTPUT_LIMIT_HEADER, OUTPUT_LIMIT_HEADER_VALUE)
    }
    return response
  }

  private fun measuredUsage(model: Model, request: ChatRequest, state: TurnState): TokenUsage? {
    val snapshot = model.runtimeHelper.readTokenUsage(model)
    return resolveTokenUsage(
      prefillTokens = snapshot?.promptTokens,
      decodeTokens = snapshot?.completionTokens,
      hadModelInput = request.hadModelInput,
      hadVisibleOutput = state.text.isNotEmpty() || state.toolCalls.isNotEmpty(),
    )
  }

  private fun measuredFinish(
    model: Model,
    request: ChatRequest,
    state: TurnState,
    usage: TokenUsage?,
  ): String? {
    return resolveFinishReason(
      sawToolCalls = state.toolCalls.isNotEmpty(),
      contextLimitTokens = contextLimit(model),
      usage = usage,
      contextLimitError = state.contextLimitError,
      maxOutputTokens = request.maxOutputTokens,
    )
  }

  private fun contextLimit(model: Model): Int =
    model.getIntConfigValue(ConfigKeys.MAX_TOKENS, DEFAULT_MAX_TOKEN)

  private fun logFinish(finish: String?, usage: TokenUsage?, limit: Int) {
    if (finish == "length") {
      Log.i(TAG, "finish_reason=length usage=$usage kvCacheLimit=$limit")
    } else if (finish == null) {
      Log.i(
        TAG,
        "finish_reason left null; LiteRT-LM 0.18.1 did not distinguish stop from length. " +
          "usage=$usage kvCacheLimit=$limit",
      )
    }
  }

  private fun requestStop(model: Model) {
    Thread({
        try {
          model.runtimeHelper.stopResponse(model)
        } catch (e: Exception) {
          Log.w(TAG, "Failed to cancel generation", e)
        }
      })
      .apply {
        isDaemon = true
        name = "openai-cancel"
        start()
      }
  }

  private fun writeSse(body: SseBody, payload: String): Boolean =
    body.write(payload.toByteArray(Charsets.UTF_8))

  /** SSE chunk carrying a complete tool call (streaming tool calling). */
  private fun toolCallChunkJson(model: String, toolCalls: List<ToolCall>, startIndex: Int): String {
    val deltas =
      toolCalls.mapIndexed { offset, tc ->
        val index = startIndex + offset
        "{\"index\":$index,\"id\":\"call_local_$index\",\"type\":\"function\"," +
          "\"function\":{\"name\":\"${escapeJson(tc.name)}\"," +
          "\"arguments\":\"${escapeJson(anyToJsonString(tc.arguments))}\"}}"
      }.joinToString(",")
    return completionChunk(
      model = model,
      deltaJson = "{\"tool_calls\":[$deltas]}",
      finishReason = null,
      usage = null,
      includeUsage = false,
    )
  }

  private fun toolCallObjects(toolCalls: List<ToolCall>): String {
    return toolCalls.mapIndexed { i, tc ->
      "{\"id\":\"call_local_$i\",\"type\":\"function\"," +
        "\"function\":{\"name\":\"${escapeJson(tc.name)}\"," +
        "\"arguments\":\"${escapeJson(anyToJsonString(tc.arguments))}\"}}"
    }.joinToString(",")
  }

  /** Serializes a plain [Map]/`List` value (from a LiteRT [ToolCall]) back into a JSON string. */
  private fun anyToJsonString(value: Any?): String {
    return when (value) {
      null -> "null"
      is String -> "\"${escapeJson(value)}\""
      is Boolean -> value.toString()
      is Int, is Long, is Double, is Float -> value.toString()
      is Map<*, *> ->
        value.entries.joinToString(",") { (k, v) ->
          "\"${escapeJson(k.toString())}\":${anyToJsonString(v)}"
        }.let { "{$it}" }
      is List<*> -> value.joinToString(",") { anyToJsonString(it) }.let { "[$it]" }
      else -> "\"${escapeJson(value.toString())}\""
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  // Helpers
  ////////////////////////////////////////////////////////////////////////////////////////////////

  private fun jsonResponse(status: Response.Status, body: String): Response {
    val res = newFixedLengthResponse(status, "application/json", body)
    res.addHeader("Access-Control-Allow-Origin", "*")
    return res
  }

  private fun jsonError(
    status: Response.Status,
    message: String,
    type: String = "invalid_request_error",
  ): Response {
    return jsonResponse(status, errorJson(message, type))
  }

  private fun escapeJson(s: String): String {
    val sb = StringBuilder(s.length + 16)
    for (c in s) {
      when (c) {
        '"' -> sb.append("\\\"")
        '\\' -> sb.append("\\\\")
        '\n' -> sb.append("\\n")
        '\r' -> sb.append("\\r")
        '\t' -> sb.append("\\t")
        '\b' -> sb.append("\\b")
        '\u000C' -> sb.append("\\f")
        else -> if (c < ' ') sb.append("\\u%04x".format(c.code)) else sb.append(c)
      }
    }
    return sb.toString()
  }
}