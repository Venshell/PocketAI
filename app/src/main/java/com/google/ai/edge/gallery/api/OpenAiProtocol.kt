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

import java.io.InputStream
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * OpenAI Chat Completions helpers with no Android or LiteRT-LM dependency.
 *
 * Token numbers are formatted only when the caller supplies counts from the runtime. Nothing in
 * this file estimates tokens from character length.
 *
 * LiteRT-LM 0.11 (the version this app ships) does not expose a per-request `max_output_tokens`
 * or a generation stop reason on [com.google.ai.edge.litertlm.Conversation]. Output caps are
 * therefore applied as a bound on streamed text pieces, and `finish_reason` is left null when the
 * runtime did not provide enough information to tell a stop token from the KV-cache limit.
 */
internal const val OUTPUT_LIMIT_HEADER = "X-PocketAI-Output-Limit"
internal const val OUTPUT_LIMIT_HEADER_VALUE =
  "max_tokens bounds streamed text pieces; litert-lm 0.11 has no max_output_tokens"

internal data class ParsedToolCall(val id: String, val name: String, val arguments: String)

internal data class ParsedMessage(
  val role: String,
  val text: String,
  val toolCallId: String? = null,
  val name: String? = null,
  val toolCalls: List<ParsedToolCall> = emptyList(),
  val rawContent: JsonElement? = null,
)

internal data class ParsedChatRequest(
  val source: JsonObject,
  val prompt: String,
  val history: List<ParsedMessage>,
  val imageDataUrls: List<String>,
  val stream: Boolean,
  val temperature: Double?,
  val topP: Double?,
  val topK: Int?,
  val seed: Int?,
  val maxOutputPieces: Int?,
  val requestedModel: String?,
  val hadModelInput: Boolean,
)

internal data class TokenUsage(val promptTokens: Int, val completionTokens: Int) {
  val totalTokens: Int = promptTokens + completionTokens
}

internal sealed class ParseOutcome {
  data class Ok(val request: ParsedChatRequest) : ParseOutcome()

  data class Rejected(val message: String) : ParseOutcome()
}

private val UNSUPPORTED_KEYS =
  setOf(
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "logit_bias",
    "top_logprobs",
    "response_format",
    "best_of",
    "suffix",
    "modalities",
    "audio",
    "prediction",
  )

internal fun parseChatBody(body: String): ParseOutcome {
  val obj =
    try {
      Json.parseToJsonElement(body).jsonObject
    } catch (e: Exception) {
      return ParseOutcome.Rejected("Invalid request body")
    }
  val messages =
    try {
      obj["messages"]?.jsonArray ?: return ParseOutcome.Rejected("Invalid request body")
    } catch (e: Exception) {
      return ParseOutcome.Rejected("Invalid request body")
    }

  val unsupported = mutableListOf<String>()
  for (key in UNSUPPORTED_KEYS) {
    if (obj.containsKey(key)) unsupported += key
  }
  val logprobs = obj["logprobs"]
  if (logprobs is JsonPrimitive && logprobs.booleanOrNull == true) {
    unsupported += "logprobs"
  }
  val nValue = obj["n"]
  if (nValue != null && nValue !is JsonNull) {
    val nPrimitive = nValue as? JsonPrimitive
    val n =
      nPrimitive?.intOrNull
        ?: nPrimitive?.doubleOrNull?.takeIf { it % 1.0 == 0.0 }?.toInt()
    if (n != 1) unsupported += "n"
  }
  val toolChoice = obj["tool_choice"]
  if (toolChoice != null && !isAutoToolChoice(toolChoice)) {
    unsupported += "tool_choice"
  }
  if (unsupported.isNotEmpty()) {
    return ParseOutcome.Rejected(
      "Unsupported parameter(s): ${unsupported.joinToString(", ")}. " +
        "LiteRT-LM 0.11 does not implement them, so this server rejects the request instead of " +
        "ignoring them."
    )
  }

  val temperature =
    readUnitDouble(obj, "temperature") { value ->
      if (value < 0.0) "temperature must be >= 0" else null
    }
  if (temperature is ReadFailure) return ParseOutcome.Rejected(temperature.message)
  val topP =
    readUnitDouble(obj, "top_p") { value ->
      if (value < 0.0 || value > 1.0) "top_p must be between 0 and 1" else null
    }
  if (topP is ReadFailure) return ParseOutcome.Rejected(topP.message)
  val topK =
    readUnitInt(obj, "top_k") { value -> if (value <= 0) "top_k must be positive" else null }
  if (topK is ReadFailure) return ParseOutcome.Rejected(topK.message)
  val seed = readUnitInt(obj, "seed") { null }
  if (seed is ReadFailure) return ParseOutcome.Rejected(seed.message)

  val maxTokens = readMaxOutputPieces(obj)
  if (maxTokens is ReadFailure) return ParseOutcome.Rejected(maxTokens.message)

  val history = mutableListOf<ParsedMessage>()
  val images = mutableListOf<String>()
  var prompt: String? = null
  try {
    for ((index, msgEl) in messages.withIndex()) {
      val msg = msgEl.jsonObject
      val role = msg["role"]?.jsonPrimitive?.takeIf { it.isString }?.content ?: continue
      val contentEl = msg["content"]
      val textParts = mutableListOf<String>()
      when (contentEl) {
        is JsonNull -> {}
        is JsonPrimitive -> {
          contentEl.contentOrNull?.let { textParts += it }
        }
        is JsonArray -> {
          for (part in contentEl) {
            val partObj = part.jsonObject
            when (partObj["type"]?.jsonPrimitive?.takeIf { it.isString }?.content) {
              "text" ->
                partObj["text"]?.jsonPrimitive?.takeIf { it.isString }?.content?.let {
                  textParts += it
                }
              "image_url" -> {
                val url =
                  partObj["image_url"]
                    ?.jsonObject
                    ?.get("url")
                    ?.jsonPrimitive
                    ?.takeIf { it.isString }
                    ?.content
                if (url != null) images += url
              }
            }
          }
        }
        else -> {}
      }
      val text = textParts.joinToString("\n")
      val isLast = index == messages.lastIndex
      when (role) {
        "user" -> {
          if (isLast) {
            prompt = text
          } else {
            history += ParsedMessage(role = "user", text = text)
          }
        }
        "system" -> history += ParsedMessage(role = "system", text = text)
        "assistant" -> {
          val calls = parseAssistantToolCalls(msg["tool_calls"])
          history += ParsedMessage(role = "assistant", text = text, toolCalls = calls)
        }
        "tool" ->
          history +=
            ParsedMessage(
              role = "tool",
              text = toolContent(contentEl),
              toolCallId = msg["tool_call_id"]?.jsonPrimitive?.takeIf { it.isString }?.content,
              name = msg["name"]?.jsonPrimitive?.takeIf { it.isString }?.content,
              rawContent = contentEl,
            )
      }
    }
  } catch (e: Exception) {
    return ParseOutcome.Rejected("Invalid request body")
  }

  val requested =
    obj["model"]?.jsonPrimitive?.takeIf { it.isString }?.content?.takeIf { it.isNotEmpty() }
  return ParseOutcome.Ok(
    ParsedChatRequest(
      source = obj,
      prompt = prompt ?: "",
      history = history,
      imageDataUrls = images,
      stream = obj["stream"]?.jsonPrimitive?.booleanOrNull ?: false,
      temperature = (temperature as ReadValue).value,
      topP = (topP as ReadValue).value,
      topK = (topK as ReadValue).value,
      seed = (seed as ReadValue).value,
      maxOutputPieces = (maxTokens as ReadValue).value,
      requestedModel = requested,
      hadModelInput = messages.isNotEmpty(),
    )
  )
}

internal fun validateRequestedModel(requested: String?, advertisedId: String): String? {
  if (requested == null) return null
  if (requested == advertisedId) return null
  return "The model `$requested` does not exist. This server is currently serving `$advertisedId`."
}

/**
 * Returns runtime token counts, or null when the runtime did not supply a usable measurement.
 * A `(0, 0)` pair together with real input or output is treated as "benchmark mode produced no
 * counts", not as a genuine empty tokenization.
 */
internal fun resolveTokenUsage(
  prefillTokens: Int?,
  decodeTokens: Int?,
  hadModelInput: Boolean,
  hadVisibleOutput: Boolean,
): TokenUsage? {
  if (prefillTokens == null || decodeTokens == null) return null
  if (prefillTokens < 0 || decodeTokens < 0) return null
  if (prefillTokens == 0 && decodeTokens == 0 && (hadModelInput || hadVisibleOutput)) return null
  return TokenUsage(promptTokens = prefillTokens, completionTokens = decodeTokens)
}

/**
 * Maps what this server can actually observe onto an OpenAI `finish_reason`.
 *
 * Returns null when the outcome would otherwise be a guess. LiteRT-LM 0.11 reports both stop-token
 * completion and KV-cache exhaustion through the same success callback; the KV-cache case is
 * recognizable only when benchmark token counts are available and their sum has reached the engine
 * context limit (`EngineConfig.maxNumTokens`).
 */
internal fun resolveFinishReason(
  sawToolCalls: Boolean,
  hitOutputPieceCap: Boolean,
  contextLimitTokens: Int?,
  usage: TokenUsage?,
  contextLimitError: Boolean,
): String? {
  if (sawToolCalls) return "tool_calls"
  if (hitOutputPieceCap || contextLimitError) return "length"
  if (usage != null && contextLimitTokens != null && contextLimitTokens > 0) {
    val consumed = usage.promptTokens + usage.completionTokens
    return if (consumed >= contextLimitTokens) "length" else "stop"
  }
  return null
}

/** Prefill rejected the prompt because it does not fit in the KV cache. */
internal fun isPromptTooLongMessage(message: String): Boolean {
  val normalized = message.lowercase()
  return "input token ids are too long" in normalized ||
    "exceeding the maximum number of tokens allowed" in normalized
}

/**
 * Decode stopped because the KV cache is full. This is distinct from a prompt that was rejected
 * before generation; that case is [isPromptTooLongMessage].
 */
internal fun isContextLimitMessage(message: String): Boolean {
  if (isPromptTooLongMessage(message)) return false
  val normalized = message.lowercase()
  return "kv-cache" in normalized ||
    "maximum number of tokens" in normalized ||
    "max number of tokens" in normalized
}

internal fun escapeJson(s: String): String {
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

internal fun errorJson(message: String, type: String = "invalid_request_error"): String =
  "{\"error\":{\"message\":\"${escapeJson(message)}\",\"type\":\"${escapeJson(type)}\"}}"

internal fun usageJson(usage: TokenUsage?): String {
  if (usage == null) return "null"
  return "{\"prompt_tokens\":${usage.promptTokens}," +
    "\"completion_tokens\":${usage.completionTokens}," +
    "\"total_tokens\":${usage.totalTokens}}"
}

internal fun textDeltaJson(content: String): String = "{\"content\":\"${escapeJson(content)}\"}"

internal fun completionChunk(
  model: String,
  deltaJson: String,
  finishReason: String?,
  usage: TokenUsage?,
  includeUsage: Boolean,
): String {
  val finishField =
    if (finishReason != null) "\"finish_reason\":\"$finishReason\"" else "\"finish_reason\":null"
  val usageField = if (includeUsage) ",\"usage\":${usageJson(usage)}" else ""
  return "{\"id\":\"chatcmpl-local\",\"object\":\"chat.completion.chunk\"," +
    "\"created\":${System.currentTimeMillis() / 1000},\"model\":\"${escapeJson(model)}\"," +
    "\"choices\":[{\"index\":0,\"delta\":$deltaJson,$finishField}]$usageField}"
}

internal fun completionJson(
  model: String,
  messageJson: String,
  finishReason: String?,
  usage: TokenUsage?,
): String {
  val finishField =
    if (finishReason != null) "\"finish_reason\":\"$finishReason\"" else "\"finish_reason\":null"
  return "{\"id\":\"chatcmpl-local\",\"object\":\"chat.completion\"," +
    "\"created\":${System.currentTimeMillis() / 1000},\"model\":\"${escapeJson(model)}\"," +
    "\"choices\":[{\"index\":0,\"message\":$messageJson,$finishField}]," +
    "\"usage\":${usageJson(usage)}}"
}

internal fun sseData(payload: String): String = "data: $payload\n\n"

internal fun sseDone(): String = "data: [DONE]\n\n"

/**
 * Bounded body for a chunked HTTP response. [write] blocks while the queue is full so a slow
 * client cannot accumulate an unbounded buffer, and returns false once [close] has been called.
 */
internal class SseBody(capacity: Int = 4) : InputStream() {
  private sealed interface Piece {
    class Data(val bytes: ByteArray) : Piece

    object End : Piece
  }

  private val queue = ArrayBlockingQueue<Piece>(capacity.coerceAtLeast(1))
  private val closed = AtomicBoolean(false)
  private val ended = AtomicBoolean(false)
  private var pending: ByteArray? = null
  private var pendingOffset = 0

  fun write(bytes: ByteArray): Boolean {
    if (bytes.isEmpty()) return !closed.get()
    if (closed.get() || ended.get()) return false
    val piece = Piece.Data(bytes)
    while (!closed.get() && !ended.get()) {
      if (queue.offer(piece, 200, TimeUnit.MILLISECONDS)) return true
    }
    return false
  }

  fun complete() {
    ended.set(true)
    queue.offer(Piece.End)
  }

  override fun read(): Int {
    val one = ByteArray(1)
    val n = read(one, 0, 1)
    return if (n < 0) -1 else one[0].toInt() and 0xff
  }

  override fun read(b: ByteArray, off: Int, len: Int): Int {
    if (len == 0) return 0
    while (true) {
      val current = pending
      if (current != null && pendingOffset < current.size) {
        val n = minOf(len, current.size - pendingOffset)
        current.copyInto(b, off, pendingOffset, pendingOffset + n)
        pendingOffset += n
        if (pendingOffset >= current.size) {
          pending = null
          pendingOffset = 0
        }
        return n
      }
      val next =
        try {
          queue.poll(200, TimeUnit.MILLISECONDS)
        } catch (e: InterruptedException) {
          Thread.currentThread().interrupt()
          return -1
        }
      when (next) {
        is Piece.Data -> {
          pending = next.bytes
          pendingOffset = 0
        }
        Piece.End -> return -1
        null -> if ((ended.get() || closed.get()) && queue.isEmpty()) return -1
      }
    }
  }

  override fun close() {
    closed.set(true)
    ended.set(true)
    // Best effort. A full queue must not block close(); read() also stops once [ended] is set
    // and the queued bytes have been drained.
    queue.offer(Piece.End)
  }
}

private sealed class ReadResult<out T>

private class ReadValue<out T>(val value: T?) : ReadResult<T>()

private class ReadFailure(val message: String) : ReadResult<Nothing>()

private fun readUnitDouble(
  obj: JsonObject,
  key: String,
  validate: (Double) -> String?,
): ReadResult<Double> {
  val el = obj[key] ?: return ReadValue(null)
  if (el is JsonNull) return ReadValue(null)
  val primitive = el as? JsonPrimitive ?: return ReadFailure("$key must be a number")
  val value = primitive.doubleOrNull ?: return ReadFailure("$key must be a number")
  validate(value)?.let {
    return ReadFailure(it)
  }
  return ReadValue(value)
}

private fun readUnitInt(
  obj: JsonObject,
  key: String,
  validate: (Int) -> String?,
): ReadResult<Int> {
  val el = obj[key] ?: return ReadValue(null)
  if (el is JsonNull) return ReadValue(null)
  val primitive = el as? JsonPrimitive ?: return ReadFailure("$key must be an integer")
  val value =
    primitive.intOrNull
      ?: primitive.doubleOrNull?.takeIf { it % 1.0 == 0.0 }?.toInt()
      ?: return ReadFailure("$key must be an integer")
  validate(value)?.let {
    return ReadFailure(it)
  }
  return ReadValue(value)
}

private fun readMaxOutputPieces(obj: JsonObject): ReadResult<Int> {
  val hasMax = obj.containsKey("max_tokens")
  val hasCompletion = obj.containsKey("max_completion_tokens")
  val maxTokens = if (hasMax) readUnitInt(obj, "max_tokens") { null } else ReadValue(null)
  if (maxTokens is ReadFailure) return maxTokens
  val maxCompletion =
    if (hasCompletion) readUnitInt(obj, "max_completion_tokens") { null } else ReadValue(null)
  if (maxCompletion is ReadFailure) return maxCompletion
  val a = (maxTokens as ReadValue).value
  val b = (maxCompletion as ReadValue).value
  if (a != null && b != null && a != b) {
    return ReadFailure("max_tokens and max_completion_tokens disagree ($a vs $b)")
  }
  val chosen = b ?: a
  if (chosen != null && chosen <= 0) {
    return ReadFailure("max_tokens must be a positive integer")
  }
  return ReadValue(chosen)
}

private fun isAutoToolChoice(element: JsonElement): Boolean {
  val primitive = element as? JsonPrimitive ?: return false
  return primitive.isString && primitive.content == "auto"
}

private fun parseAssistantToolCalls(element: JsonElement?): List<ParsedToolCall> {
  val array = element as? JsonArray ?: return emptyList()
  val calls = mutableListOf<ParsedToolCall>()
  for (tc in array) {
    val tcObj = tc.jsonObject
    val id = tcObj["id"]?.jsonPrimitive?.takeIf { it.isString }?.content ?: ""
    val fnObj = tcObj["function"]?.jsonObject
    val name = fnObj?.get("name")?.jsonPrimitive?.takeIf { it.isString }?.content ?: ""
    val args = fnObj?.get("arguments")?.jsonPrimitive?.takeIf { it.isString }?.content ?: "{}"
    if (name.isNotEmpty()) calls += ParsedToolCall(id = id, name = name, arguments = args)
  }
  return calls
}

private fun toolContent(contentEl: JsonElement?): String {
  return when (contentEl) {
    is JsonPrimitive -> contentEl.content
    null -> ""
    else -> contentEl.toString()
  }
}
