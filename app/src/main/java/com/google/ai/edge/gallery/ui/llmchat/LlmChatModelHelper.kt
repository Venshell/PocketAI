/*
 * Copyright 2025 Google LLC
 *
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

package com.google.ai.edge.gallery.ui.llmchat

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.ai.edge.gallery.common.cleanUpMediapipeTaskErrorMessage
import com.google.ai.edge.gallery.data.Accelerator
import com.google.ai.edge.gallery.data.ConfigKeys
import com.google.ai.edge.gallery.data.DEFAULT_MAX_TOKEN
import com.google.ai.edge.gallery.data.DEFAULT_TEMPERATURE
import com.google.ai.edge.gallery.data.DEFAULT_TOPK
import com.google.ai.edge.gallery.data.DEFAULT_TOPP
import com.google.ai.edge.gallery.data.DEFAULT_VISION_ACCELERATOR
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.ModelCapability
import com.google.ai.edge.gallery.data.THOUGHT_CHANNEL
import com.google.ai.edge.gallery.data.markInitializationFailed
import com.google.ai.edge.gallery.data.markInitializationStarted
import com.google.ai.edge.gallery.data.markInitialized
import com.google.ai.edge.gallery.data.resetInitialization
import com.google.ai.edge.gallery.runtime.CleanUpListener
import com.google.ai.edge.gallery.runtime.LlmModelHelper
import com.google.ai.edge.gallery.runtime.ResultListener
import com.google.ai.edge.gallery.runtime.SamplerOverride
import com.google.ai.edge.gallery.runtime.TokenUsageSnapshot
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.RepetitionPenaltyConfig
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ToolCall
import com.google.ai.edge.litertlm.ToolProvider
import java.io.ByteArrayOutputStream
import java.util.concurrent.CancellationException
import kotlinx.coroutines.CoroutineScope

private const val TAG = "AGLlmChatModelHelper"

data class LlmModelInstance(val engine: Engine, var conversation: Conversation)

object LlmChatModelHelper : LlmModelHelper {
  // Indexed by model name.
  private val cleanUpListeners: MutableMap<String, CleanUpListener> = mutableMapOf()

  @OptIn(ExperimentalApi::class) // opt-in experimental flags
  override fun initialize(
    context: Context,
    model: Model,
    taskId: String,
    supportImage: Boolean,
    supportAudio: Boolean,
    onDone: (String) -> Unit,
    systemInstruction: Contents?,
    tools: List<ToolProvider>,
    enableConversationConstrainedDecoding: Boolean,
    coroutineScope: CoroutineScope?,
  ) {
    if (model.instance != null) {
      Log.d(TAG, "Model '${model.name}' already initialized in LlmChatModelHelper. Skipping.")
      model.markInitialized()
      onDone("")
      return
    }
    model.markInitializationStarted()
    // Prepare options.
    val maxTokens =
      model.getIntConfigValue(key = ConfigKeys.MAX_TOKENS, defaultValue = DEFAULT_MAX_TOKEN)
    val topK = model.getIntConfigValue(key = ConfigKeys.TOPK, defaultValue = DEFAULT_TOPK)
    val topP = model.getFloatConfigValue(key = ConfigKeys.TOPP, defaultValue = DEFAULT_TOPP)
    val temperature =
      model.getFloatConfigValue(key = ConfigKeys.TEMPERATURE, defaultValue = DEFAULT_TEMPERATURE)
    val accelerator =
      model.getStringConfigValue(key = ConfigKeys.ACCELERATOR, defaultValue = Accelerator.GPU.label)
    val visionAccelerator =
      model.getStringConfigValue(
        key = ConfigKeys.VISION_ACCELERATOR,
        defaultValue = DEFAULT_VISION_ACCELERATOR.label,
      )
    val visionBackend =
      when (visionAccelerator) {
        Accelerator.CPU.label -> Backend.CPU()
        Accelerator.GPU.label -> Backend.GPU()
        Accelerator.NPU.label ->
          Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
        Accelerator.TPU.label ->
          Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
        else -> Backend.GPU()
      }
    val shouldEnableImage = supportImage
    val shouldEnableAudio = supportAudio
    val preferredBackend =
      when (accelerator) {
        Accelerator.CPU.label -> Backend.CPU()
        Accelerator.GPU.label -> Backend.GPU()
        Accelerator.NPU.label ->
          Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
        Accelerator.TPU.label ->
          Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
        else -> Backend.CPU()
      }
    Log.d(TAG, "Preferred backend: $preferredBackend")

    val modelPath = model.getPath(context = context)
    val engineConfig =
      EngineConfig(
        modelPath = modelPath,
        backend = preferredBackend,
        visionBackend = if (shouldEnableImage) visionBackend else null, // must be GPU for Gemma 3n
        audioBackend = if (shouldEnableAudio) Backend.CPU() else null, // must be CPU for Gemma 3n
        maxNumTokens = maxTokens,
        cacheDir =
          if (modelPath.startsWith("/data/local/tmp"))
            context.getExternalFilesDir(null)?.absolutePath
          else null,
      )

    // Check if the model file supports speculative decoding.
    var supportsSpeculativeDecoding = false
    // Check if the model file supports speculative decoding.
    try {
      com.google.ai.edge.litertlm.ModelInfo.from(modelPath).use { modelInfo ->
        if (modelInfo is com.google.ai.edge.litertlm.ModelInfo.Llm) {
          supportsSpeculativeDecoding = modelInfo.hasSpeculativeDecodingSupport()
        }
      }
    } catch (e: Exception) {
      // Ignore exceptions and assume not supported.
    }
    // Create an instance of LiteRT LM engine and conversation.
    try {
      // Benchmark accounting is captured when the native engine is created. It is what exposes
      // lastPrefillTokenCount / lastDecodeTokenCount. Sampling itself is unchanged.
      val previousBenchmark = ExperimentalFlags.enableBenchmark
      var speculativeDecoding = false
      // Check if the model supports speculative decoding for the given task type and if the
      // speculative decoding is enabled in the settings.
      if (
        supportsSpeculativeDecoding &&
          model.capabilityToTaskTypes[ModelCapability.SPECULATIVE_DECODING]?.contains(taskId) ==
            true
      ) {
        speculativeDecoding =
          model.getBooleanConfigValue(
            key = ConfigKeys.ENABLE_SPECULATIVE_DECODING,
            defaultValue = false,
          )
      }
      Log.d(TAG, "Speculative decoding enabled: $speculativeDecoding")
      val engine =
        try {
          ExperimentalFlags.enableBenchmark = true
          ExperimentalFlags.enableSpeculativeDecoding = speculativeDecoding
          Engine(engineConfig).also { it.initialize() }
        } finally {
          ExperimentalFlags.enableBenchmark = previousBenchmark
          ExperimentalFlags.enableSpeculativeDecoding = false
        }

      ExperimentalFlags.enableConversationConstrainedDecoding =
        enableConversationConstrainedDecoding
      val conversation =
        engine.createConversation(
          ConversationConfig(
            samplerConfig =
              if (preferredBackend is Backend.NPU) {
                null
              } else {
                SamplerConfig(
                  topK = topK,
                  topP = topP.toDouble(),
                  temperature = temperature.toDouble(),
                )
              },
            systemInstruction = systemInstruction,
            tools = tools,
          )
        )
      ExperimentalFlags.enableConversationConstrainedDecoding = false
      model.instance = LlmModelInstance(engine = engine, conversation = conversation)
    } catch (e: Exception) {
      val errorMsg = cleanUpMediapipeTaskErrorMessage(e.message ?: "Unknown error")
      model.markInitializationFailed(errorMsg)
      onDone(errorMsg)
      return
    }
    model.markInitialized()
    onDone("")
  }

  @OptIn(ExperimentalApi::class) // opt-in experimental flags
  override fun resetConversation(
    model: Model,
    supportImage: Boolean,
    supportAudio: Boolean,
    systemInstruction: Contents?,
    tools: List<ToolProvider>,
    automaticToolCalling: Boolean,
    enableConversationConstrainedDecoding: Boolean,
    initialMessages: List<Message>,
    samplerOverride: SamplerOverride?,
  ) {
    Log.d(TAG, "Resetting conversation for model '${model.name}'")

    val instance =
      model.instance as LlmModelInstance?
        ?: throw IllegalStateException("Model '${model.name}' is not initialized")
    val accelerator =
      model.getStringConfigValue(
        key = ConfigKeys.ACCELERATOR,
        defaultValue = Accelerator.GPU.label,
      )
    val npuBackend = accelerator == Accelerator.NPU.label || accelerator == Accelerator.TPU.label
    if (npuBackend && samplerOverride != null) {
      throw IllegalArgumentException(
        "temperature, top_p, top_k, and seed cannot be applied when the accelerator is " +
          "$accelerator. Gallery 1.0.23 still omits SamplerConfig on NPU and TPU."
      )
    }
    // close() throws if it is called twice. A previous failed reset may already have closed it,
    // and the replacement still has to be created so the next request is not stuck on a dead
    // conversation.
    try {
      instance.conversation.close()
    } catch (e: Exception) {
      Log.w(TAG, "Previous conversation could not be closed cleanly", e)
    }

    val engine = instance.engine
    val topK = samplerOverride?.topK
      ?: model.getIntConfigValue(key = ConfigKeys.TOPK, defaultValue = DEFAULT_TOPK)
    val topP = samplerOverride?.topP
      ?: model.getFloatConfigValue(key = ConfigKeys.TOPP, defaultValue = DEFAULT_TOPP).toDouble()
    val temperature = samplerOverride?.temperature
      ?: model.getFloatConfigValue(key = ConfigKeys.TEMPERATURE, defaultValue = DEFAULT_TEMPERATURE)
        .toDouble()
    val seed = samplerOverride?.seed ?: 0
    val shouldEnableImage = supportImage
    val shouldEnableAudio = supportAudio
    Log.d(TAG, "Enable image: $shouldEnableImage, enable audio: $shouldEnableAudio")

    try {
      ExperimentalFlags.enableConversationConstrainedDecoding =
        enableConversationConstrainedDecoding
      val newConversation =
        engine.createConversation(
          ConversationConfig(
            samplerConfig =
              if (npuBackend) {
                null
              } else {
                SamplerConfig(topK = topK, topP = topP, temperature = temperature, seed = seed)
              },
            systemInstruction = systemInstruction,
            tools = tools,
            automaticToolCalling = automaticToolCalling,
            initialMessages = initialMessages,
          )
        )
      instance.conversation = newConversation
      Log.d(TAG, "Resetting done")
    } catch (e: IllegalArgumentException) {
      throw e
    } catch (e: Exception) {
      Log.w(TAG, "Failed to reset conversation", e)
      throw IllegalStateException("Failed to rebuild conversation: ${e.message}", e)
    } finally {
      ExperimentalFlags.enableConversationConstrainedDecoding = false
    }
  }

  @OptIn(ExperimentalApi::class) // getBenchmarkInfo is experimental in LiteRT-LM 0.18.1
  override fun readTokenUsage(model: Model): TokenUsageSnapshot? {
    val instance = model.instance as? LlmModelInstance ?: return null
    return try {
      val info = instance.conversation.getBenchmarkInfo()
      TokenUsageSnapshot(
        promptTokens = info.lastPrefillTokenCount,
        completionTokens = info.lastDecodeTokenCount,
      )
    } catch (e: Exception) {
      Log.w(TAG, "Native token counts are unavailable", e)
      null
    }
  }

  override fun cleanUp(model: Model, onDone: () -> Unit) {
    if (model.instance == null) {
      return
    }

    val instance = model.instance as LlmModelInstance

    try {
      instance.conversation.close()
    } catch (e: Exception) {
      Log.e(TAG, "Failed to close the conversation: ${e.message}")
    }

    try {
      instance.engine.close()
    } catch (e: Exception) {
      Log.e(TAG, "Failed to close the engine: ${e.message}")
    }

    val onCleanUp = cleanUpListeners.remove(model.name)
    if (onCleanUp != null) {
      onCleanUp()
    }
    model.resetInitialization()

    onDone()
    Log.d(TAG, "Clean up done.")
  }

  override fun stopResponse(model: Model) {
    val instance = model.instance as? LlmModelInstance ?: return
    try {
      instance.conversation.cancelProcess()
    } catch (e: IllegalStateException) {
      Log.w(TAG, "Conversation is not alive, cannot cancel process", e)
    }
  }

  override fun runInference(
    model: Model,
    input: String,
    resultListener: ResultListener,
    cleanUpListener: CleanUpListener,
    onError: (message: String) -> Unit,
    images: List<Bitmap>,
    audioClips: List<ByteArray>,
    coroutineScope: CoroutineScope?,
    extraContext: Map<String, String>?,
    onToolCall: (List<ToolCall>) -> Unit,
    maxOutputTokens: Int?,
    repetitionPenalty: Float?,
    presencePenalty: Float?,
    frequencyPenalty: Float?,
  ) {
    val instance = model.instance as? LlmModelInstance
    if (instance == null) {
      onError("LlmModelInstance is not initialized.")
      return
    }

    // Set listener.
    if (!cleanUpListeners.containsKey(model.name)) {
      cleanUpListeners[model.name] = cleanUpListener
    }

    val conversation = instance.conversation

    val contents = mutableListOf<Content>()
    for (image in images) {
      contents.add(Content.ImageBytes(image.toPngByteArray()))
    }
    for (audioClip in audioClips) {
      contents.add(Content.AudioBytes(audioClip))
    }
    // add the text after image and audio for the accurate last token
    if (input.trim().isNotEmpty()) {
      contents.add(Content.Text(input))
    }

    // Set enable_thinking to false by default using boolean literals for proper JSON serialization.
    val enableThinking = extraContext?.get("enable_thinking") == "true"
    val finalExtraContext: Map<String, Any> =
      (extraContext ?: emptyMap()) + ("enable_thinking" to enableThinking)

    val penaltyConfig =
      if (repetitionPenalty != null || presencePenalty != null || frequencyPenalty != null) {
        RepetitionPenaltyConfig(
          repetitionPenalty = repetitionPenalty,
          presencePenalty = presencePenalty,
          frequencyPenalty = frequencyPenalty,
        )
      } else {
        null
      }
    conversation.sendMessageAsync(
      Contents.of(contents),
      object : MessageCallback {
        override fun onMessage(message: Message) {
          // When automatic tool calling is disabled, the conversation surfaces tool calls
          // (instead of executing them). Forward them to the caller so it can hand them back to
          // the client (e.g. the OpenAI API server) for client-driven execution.
          if (message.toolCalls.isNotEmpty()) {
            onToolCall(message.toolCalls)
          }
          resultListener(message.toString(), false, message.channels[THOUGHT_CHANNEL])
        }

        override fun onDone() {
          resultListener("", true, null)
        }

        override fun onError(throwable: Throwable) {
          if (throwable is CancellationException) {
            Log.i(TAG, "The inference is cancelled.")
            resultListener("", true, null)
          } else {
            Log.e(TAG, "onError", throwable)
            onError("Error: ${throwable.message}")
          }
        }
      },
      finalExtraContext,
      repetitionPenaltyConfig = penaltyConfig,
      maxOutputToken = maxOutputTokens,
    )
  }

  private fun Bitmap.toPngByteArray(): ByteArray {
    val stream = ByteArrayOutputStream()
    this.compress(Bitmap.CompressFormat.PNG, 100, stream)
    return stream.toByteArray()
  }
}
