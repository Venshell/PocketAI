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

import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class OpenAiProtocolTest {
  @Test
  fun parsesChineseEnglishAndMixedHistoryWithoutEstimatingTokens() {
    val body =
      """
      {"model":"gemma","messages":[
        {"role":"system","content":"只用中文回答"},
        {"role":"user","content":"Hello 你好"},
        {"role":"assistant","content":"你好"},
        {"role":"user","content":"mixed 混合 text"}
      ]}
      """
        .trimIndent()
    val request = (parseChatBody(body) as ParseOutcome.Ok).request
    assertEquals("mixed 混合 text", request.prompt)
    assertEquals(listOf("system", "user", "assistant"), request.history.map { it.role })
    assertEquals("只用中文回答", request.history[0].text)
    assertEquals("Hello 你好", request.history[1].text)
    assertNull(request.maxOutputTokens)
    assertNull(request.temperature)
  }

  @Test
  fun keepsInterleavedSystemMessagesInOrder() {
    val body =
      """
      {"messages":[
        {"role":"system","content":"A"},
        {"role":"user","content":"u1"},
        {"role":"system","content":"B"},
        {"role":"assistant","content":"m"},
        {"role":"user","content":"u2"}
      ]}
      """
        .trimIndent()
    val history = (parseChatBody(body) as ParseOutcome.Ok).request.history
    assertEquals(listOf("system", "user", "system", "assistant"), history.map { it.role })
    assertEquals(listOf("A", "u1", "B", "m"), history.map { it.text })
  }

  @Test
  fun acceptsNEqualsOneAndRejectsHigherN() {
    val one =
      parseChatBody("""{"messages":[{"role":"user","content":"hi"}],"n":1}""")
    assertTrue(one is ParseOutcome.Ok)
    val two =
      parseChatBody("""{"messages":[{"role":"user","content":"hi"}],"n":2}""")
    assertTrue((two as ParseOutcome.Rejected).message.contains("n"))
  }

  @Test
  fun appliesPenaltiesThatLiteRtLm0181CanPassThrough() {
    val body =
      """
      {"messages":[{"role":"user","content":"hi"}],
       "repetition_penalty":1.2,"presence_penalty":0.4,"frequency_penalty":-0.2}
      """
        .trimIndent()
    val request = (parseChatBody(body) as ParseOutcome.Ok).request
    assertEquals(1.2f, request.repetitionPenalty!!, 0.0001f)
    assertEquals(0.4f, request.presencePenalty!!, 0.0001f)
    assertEquals(-0.2f, request.frequencyPenalty!!, 0.0001f)
    val low =
      parseChatBody("""{"messages":[{"role":"user","content":"hi"}],"repetition_penalty":0.5}""")
    assertTrue((low as ParseOutcome.Rejected).message.contains("repetition_penalty"))
  }

  @Test
  fun rejectsUnsupportedStopInsteadOfIgnoringIt() {
    val outcome = parseChatBody("""{"messages":[{"role":"user","content":"hi"}],"stop":["END"]}""")
    assertTrue(outcome is ParseOutcome.Rejected)
    assertTrue((outcome as ParseOutcome.Rejected).message.contains("stop"))
  }

  @Test
  fun appliesTemperatureTopPAndMaxTokensWhenTheyAreValid() {
    val body =
      """
      {"messages":[{"role":"user","content":"hi"}],
       "temperature":0.2,"top_p":0.5,"top_k":8,"seed":7,
       "max_completion_tokens":3}
      """
        .trimIndent()
    val request = (parseChatBody(body) as ParseOutcome.Ok).request
    assertEquals(0.2, request.temperature!!, 0.0)
    assertEquals(0.5, request.topP!!, 0.0)
    assertEquals(8, request.topK)
    assertEquals(7, request.seed)
    assertEquals(3, request.maxOutputTokens)
  }

  @Test
  fun rejectsDisagreeingOutputCapsAndNonPositiveCaps() {
    val disagree =
      parseChatBody(
        """{"messages":[{"role":"user","content":"hi"}],"max_tokens":2,"max_completion_tokens":4}"""
      )
    assertTrue((disagree as ParseOutcome.Rejected).message.contains("disagree"))
    val zero =
      parseChatBody("""{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}""")
    assertTrue((zero as ParseOutcome.Rejected).message.contains("positive"))
    val negativeTemp =
      parseChatBody("""{"messages":[{"role":"user","content":"hi"}],"temperature":-0.1}""")
    assertTrue(negativeTemp is ParseOutcome.Rejected)
  }

  @Test
  fun modelIdMustMatchTheAdvertisedId() {
    assertNull(validateRequestedModel(null, "Gemma-4-E2B"))
    assertNull(validateRequestedModel("Gemma-4-E2B", "Gemma-4-E2B"))
    val error = validateRequestedModel("missing", "Gemma-4-E2B")
    assertNotNull(error)
    assertTrue(error!!.contains("missing"))
    assertTrue(error.contains("Gemma-4-E2B"))
  }

  @Test
  fun usageUsesRuntimeCountsForChineseAndDoesNotFallBackToLengthOverFour() {
    val prompt = "你好世界"
    val completion = "Hello 混合"
    val charEstimate = (prompt.length + completion.length) / 4
    val usage =
      resolveTokenUsage(
        prefillTokens = 8,
        decodeTokens = 5,
        hadModelInput = true,
        hadVisibleOutput = true,
      )
    assertEquals(TokenUsage(8, 5), usage)
    assertTrue(usage!!.totalTokens != charEstimate)
    assertNull(
      resolveTokenUsage(
        prefillTokens = 0,
        decodeTokens = 0,
        hadModelInput = true,
        hadVisibleOutput = true,
      )
    )
    assertNull(resolveTokenUsage(null, null, hadModelInput = true, hadVisibleOutput = false))
    val json = completionJson("gemma", "{\"role\":\"assistant\",\"content\":\"$prompt\"}", "stop", usage)
    assertTrue(json.contains("\"prompt_tokens\":8"))
    assertTrue(json.contains("\"completion_tokens\":5"))
    assertFalse(json.contains("\"prompt_tokens\":${prompt.length / 4}"))
  }

  @Test
  fun finishReasonDistinguishesLengthStopToolsAndUnknown() {
    val underLimit = TokenUsage(promptTokens = 100, completionTokens = 20)
    assertEquals(
      "stop",
      resolveFinishReason(
        sawToolCalls = false,
        contextLimitTokens = 4000,
        usage = underLimit,
        contextLimitError = false,
      ),
    )
    assertEquals(
      "length",
      resolveFinishReason(
        sawToolCalls = false,
        contextLimitTokens = 120,
        usage = underLimit,
        contextLimitError = false,
      ),
    )
    assertEquals(
      "length",
      resolveFinishReason(
        sawToolCalls = false,
        contextLimitTokens = 4000,
        usage = underLimit,
        contextLimitError = false,
        maxOutputTokens = 20,
      ),
    )
    assertEquals(
      "tool_calls",
      resolveFinishReason(
        sawToolCalls = true,
        contextLimitTokens = 4000,
        usage = underLimit,
        contextLimitError = false,
        maxOutputTokens = 20,
      ),
    )
    assertNull(
      resolveFinishReason(
        sawToolCalls = false,
        contextLimitTokens = 4000,
        usage = null,
        contextLimitError = false,
        maxOutputTokens = 8,
      )
    )
    assertEquals(
      "length",
      resolveFinishReason(
        sawToolCalls = false,
        contextLimitTokens = null,
        usage = null,
        contextLimitError = true,
      ),
    )
    assertTrue(isContextLimitMessage("Maximum kv-cache size reached."))
    assertTrue(isContextLimitMessage("Reached maximum number of tokens."))
    assertFalse(isContextLimitMessage("model file is missing"))
    assertTrue(
      isPromptTooLongMessage(
        "Input token ids are too long. Exceeding the maximum number of tokens allowed: 5000 >= 4000"
      )
    )
    assertFalse(
      isContextLimitMessage(
        "Input token ids are too long. Exceeding the maximum number of tokens allowed: 5000 >= 4000"
      )
    )
  }

  @Test
  fun finalSseChunkCarriesUsageAndDoneMarkerWhileDeltasDoNot() {
    val delta = sseData(completionChunk("gemma", textDeltaJson("你"), null, null, false))
    assertTrue(delta.startsWith("data: "))
    assertTrue(delta.contains("你"))
    assertFalse(delta.contains("usage"))
    assertTrue(delta.endsWith("\n\n"))
    val usage = TokenUsage(3, 1)
    val end = sseData(completionChunk("gemma", "{}", "length", usage, includeUsage = true))
    assertTrue(end.contains("\"finish_reason\":\"length\""))
    assertTrue(end.contains("\"prompt_tokens\":3"))
    assertEquals("data: [DONE]\n\n", sseDone())
  }

  @Test
  fun sseBodyYieldsTheFirstChunkBeforeTheProducerFinishes() {
    val body = SseBody(capacity = 1)
    val firstQueued = CountDownLatch(1)
    val secondQueued = CountDownLatch(1)
    val writer =
      Thread {
        assertTrue(body.write("data: first\n\n".toByteArray()))
        firstQueued.countDown()
        assertTrue(body.write("data: second\n\n".toByteArray()))
        secondQueued.countDown()
        body.complete()
      }
    writer.start()
    assertTrue(firstQueued.await(2, TimeUnit.SECONDS))
    Thread.sleep(200)
    assertEquals(1L, secondQueued.count)
    val buffer = ByteArray(64)
    val first = body.read(buffer)
    assertTrue(first > 0)
    assertEquals("data: first\n\n", String(buffer, 0, first))
    assertTrue(secondQueued.await(2, TimeUnit.SECONDS))
    val rest = StringBuilder()
    while (true) {
      val n = body.read(buffer)
      if (n < 0) break
      rest.append(String(buffer, 0, n))
    }
    assertEquals("data: second\n\n", rest.toString())
    writer.join(2000)
    assertFalse(writer.isAlive)
  }

  @Test
  fun sseBodyWriteFailsAfterCloseSoTheProducerStops() {
    val body = SseBody(capacity = 1)
    body.close()
    assertFalse(body.write("data: late\n\n".toByteArray()))
  }
}
