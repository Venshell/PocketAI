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

import java.util.Properties

plugins {
  alias(libs.plugins.android.application)
  // Note: set apply to true to enable google-services (requires google-services.json).
  alias(libs.plugins.google.services) apply false
  alias(libs.plugins.kotlin.compose)
  alias(libs.plugins.kotlin.serialization)
  alias(libs.plugins.protobuf)
  alias(libs.plugins.hilt.application)
  alias(libs.plugins.oss.licenses)
  alias(libs.plugins.ksp)
  alias(libs.plugins.legacy.kapt)
}

android {
  namespace = "com.google.ai.edge.gallery"
  compileSdk { this.version = release(37) { minorApiLevel = 0 } }

  defaultConfig {
    applicationId = "com.google.aiedge.gallery"
    minSdk = 31
    targetSdk = 37
    versionCode = 41
    versionName = "1.3.0"

    // Needed for HuggingFace auth workflows.
    // Use the scheme of the "Redirect URLs" in HuggingFace app.
    manifestPlaceholders["appAuthRedirectScheme"] =
        "REPLACE_WITH_YOUR_REDIRECT_SCHEME_IN_HUGGINGFACE_APP"
    manifestPlaceholders["applicationName"] = "com.google.ai.edge.gallery.GalleryApplication"
    manifestPlaceholders["appIcon"] = "@mipmap/ic_launcher"

    buildConfigField("String", "FEEDBACK_API_KEY", "\"\"")

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
  }

  flavorDimensions += "device"

  productFlavors {
    // Standard build: targets current Android (minSdk 31 / Android 12+).
    create("standard") {
      dimension = "device"
    }
    // Legacy build: lowers minSdk so the app can install on older devices
    // (e.g. Android 10, which reports API 29). Built explicitly with the
    // `legacyRelease` / `legacyDebug` task variants.
    create("legacy") {
      dimension = "device"
      minSdk = 29
      applicationIdSuffix = ".legacy"
    }
  }

  // A release keystore lives outside the repo (local/keystore). CI and fresh clones
  // do not have one. AGP 9 refuses to package a release variant whose signing config
  // is missing storeFile, so only attach that config when the keystore is actually present.
  val releaseKeystore =
      run {
        val propsFile = rootProject.file("local/keystore/keystore.properties")
        if (!propsFile.exists()) {
          null
        } else {
          val props = Properties().apply { propsFile.inputStream().use { load(it) } }
          val store =
              props.getProperty("storeFile")?.let { rootProject.file("local/keystore/$it") }
          if (store != null && store.isFile) props to store else null
        }
      }

  signingConfigs {
    if (releaseKeystore != null) {
      val (props, store) = releaseKeystore
      create("release") {
        storeFile = store
        storePassword = props.getProperty("storePassword")
        keyAlias = props.getProperty("keyAlias")
        keyPassword = props.getProperty("keyPassword")
      }
    }
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
      signingConfig =
          if (releaseKeystore != null) {
            signingConfigs.getByName("release")
          } else {
            // Same fallback CI used before the AGP 9 upgrade: a debug-signed release APK
            // that can still be installed from the workflow artifact.
            signingConfigs.getByName("debug")
          }
    }
  }
  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
  }
  buildFeatures {
    compose = true
    buildConfig = true
  }
}

kotlin {
  compilerOptions {
    // `-Xcontext-receivers` was removed: experimental context receivers are superseded by
    // context parameters in newer Kotlin versions, and no code in this project uses them.
  }
}

dependencies {
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.lifecycle.runtime.ktx)
  implementation(libs.androidx.activity.compose)
  implementation(platform(libs.androidx.compose.bom))
  implementation(libs.androidx.ui)
  implementation(libs.androidx.ui.graphics)
  implementation(libs.androidx.ui.tooling.preview)
  implementation(libs.androidx.material3)
  implementation(libs.androidx.compose.navigation)
  implementation(libs.kotlinx.serialization.json)
  implementation(libs.kotlin.reflect)
  implementation(libs.material.icon.extended)
  implementation(libs.androidx.work.runtime)
  implementation(libs.androidx.datastore)
  implementation(libs.com.google.code.gson)
  implementation(libs.androidx.lifecycle.process)
  implementation(libs.androidx.security.crypto)
  implementation(libs.androidx.webkit)
  implementation(libs.litertlm)
  implementation(libs.commonmark)
  implementation(libs.richtext)
  implementation(libs.tflite)
  implementation(libs.tflite.gpu)
  implementation(libs.tflite.support)
  implementation(libs.camerax.core)
  implementation(libs.camerax.camera2)
  implementation(libs.camerax.lifecycle)
  implementation(libs.camerax.view)
  implementation(libs.openid.appauth)
  implementation(libs.androidx.splashscreen)
  implementation(libs.protobuf.javalite)
  implementation(libs.protobuf.kotlin.lite)
  implementation(libs.hilt.android)
  implementation(libs.hilt.navigation.compose)
  implementation(libs.play.services.oss.licenses)
  implementation(platform(libs.firebase.bom))
  implementation(libs.firebase.analytics)
  implementation(libs.firebase.messaging)
  implementation(libs.androidx.exifinterface)
  implementation(libs.moshi.kotlin)
  kapt(libs.hilt.android.compiler)
  testImplementation(libs.junit)
  androidTestImplementation(libs.androidx.junit)
  androidTestImplementation(libs.androidx.espresso.core)
  androidTestImplementation(platform(libs.androidx.compose.bom))
  androidTestImplementation(libs.androidx.ui.test.junit4)
  androidTestImplementation(libs.hilt.android.testing)
  debugImplementation(libs.androidx.ui.tooling)
  debugImplementation(libs.androidx.ui.test.manifest)
  ksp(libs.moshi.kotlin.codegen)
  implementation(libs.mlkit.genai.prompt)
  implementation(libs.mcp.kotlin.sdk)
  implementation(libs.ktor.client.android)
  implementation(libs.ktor.client.core)
  implementation(libs.nanohttpd)
}

protobuf {
  protoc { artifact = "com.google.protobuf:protoc:4.26.1" }
  generateProtoTasks {
    all().forEach { task ->
      task.builtins {
        create("java") { option("lite") }
        create("kotlin") { option("lite") }
      }
    }
  }
}
