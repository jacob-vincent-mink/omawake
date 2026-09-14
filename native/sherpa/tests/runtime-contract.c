#define _XOPEN_SOURCE 700

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

static void require(int condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "runtime contract failure: %s\n", message);
    exit(1);
  }
}

static void require_error_contains(const char *text) {
  const char *error = SherpaOnnxOrtRuntimeGetLastError(NULL);
  require(error != NULL && strstr(error, text) != NULL, text);
}

typedef struct {
  SherpaOnnxOrtRuntime *runtime;
  int null_registration;
  const char *expected;
} ErrorThread;

static pthread_barrier_t error_barrier;

static void *set_and_read_thread_error(void *opaque) {
  ErrorThread *thread = opaque;
  if (thread->null_registration) {
    require(!SherpaOnnxOrtRuntimeRegisterExecutionProviderLibrary(
                NULL, "invalid", "/missing/provider.so"),
            "threaded null registration must fail");
  } else {
    require(!SherpaOnnxOrtRuntimeHasExecutionProviderDevice(
                thread->runtime, "MissingExecutionProvider", "tpu"),
            "threaded invalid device type must fail");
  }
  const int wait = pthread_barrier_wait(&error_barrier);
  require(wait == 0 || wait == PTHREAD_BARRIER_SERIAL_THREAD,
          "thread error barrier must complete");
  require_error_contains(thread->expected);
  return NULL;
}

static void error_isolation_contract(void) {
  SherpaOnnxOrtRuntime *runtime = SherpaOnnxCreateOrtRuntime();
  require(runtime != NULL, "thread error runtime must be created");
  require(pthread_barrier_init(&error_barrier, NULL, 2) == 0,
          "thread error barrier must initialize");
  ErrorThread errors[2] = {
      {runtime, 0, "device_type must be cpu, gpu, npu, or empty"},
      {runtime, 1, "runtime is null"},
  };
  pthread_t threads[2];
  for (int i = 0; i != 2; ++i) {
    require(pthread_create(&threads[i], NULL, set_and_read_thread_error,
                           &errors[i]) == 0,
            "error thread must start");
  }
  for (int i = 0; i != 2; ++i) {
    require(pthread_join(threads[i], NULL) == 0, "error thread must join");
  }
  require(pthread_barrier_destroy(&error_barrier) == 0,
          "thread error barrier must be destroyed");
  SherpaOnnxDestroyOrtRuntime(runtime);
}

static void baseline_contract(void) {
  require(SherpaOnnxGetExtendedApiVersion() == 1,
          "extended API version must be 1");
  require(!SherpaOnnxOrtRuntimeRegisterExecutionProviderLibrary(
              NULL, "invalid", "/missing/provider.so"),
          "null runtime registration must fail");
  require_error_contains("runtime is null");

  SherpaOnnxOrtRuntime *first = SherpaOnnxCreateOrtRuntime();
  SherpaOnnxOrtRuntime *second = SherpaOnnxCreateOrtRuntime();
  require(first != NULL && second != NULL, "runtime creation must succeed");
  require(!SherpaOnnxOrtRuntimeRegisterExecutionProviderLibrary(
              first, "missing-provider", "/missing/provider.so"),
          "missing provider must fail");
  require(strlen(SherpaOnnxOrtRuntimeGetLastError(first)) > 0,
          "provider load failure must retain an error");
  require(!SherpaOnnxOrtRuntimeHasExecutionProviderDevice(first, "", "gpu"),
          "empty EP name must fail");
  require_error_contains("ep_name is required");
  require(!SherpaOnnxOrtRuntimeHasExecutionProviderDevice(
              first, "MissingExecutionProvider", "tpu"),
          "unknown device type must fail");
  require_error_contains("device_type must be cpu, gpu, npu, or empty");
  SherpaOnnxDestroyOrtRuntime(first);
  SherpaOnnxDestroyOrtRuntime(second);

  first = SherpaOnnxCreateOrtRuntime();
  require(first != NULL, "runtime recreation after teardown must succeed");
  SherpaOnnxDestroyOrtRuntime(first);
}

static void plugin_contract(int argc, char **argv) {
  if (argc == 1) return;
  require(argc == 11,
          "plugin mode needs library, registration, EP, device, provider config, encoder, decoder, joiner, tokens, and keywords");

  const char *library = argv[1];
  const char *registration = argv[2];
  const char *ep = argv[3];
  const char *device = argv[4];
  const char *provider_config = argv[5];
  char conflicting_path[4096];
  require(snprintf(conflicting_path, sizeof(conflicting_path), "%s.different",
                   library) < (int)sizeof(conflicting_path),
          "provider path is too long");

  SherpaOnnxOrtRuntime *first = SherpaOnnxCreateOrtRuntime();
  SherpaOnnxOrtRuntime *second = SherpaOnnxCreateOrtRuntime();
  require(first != NULL && second != NULL, "plugin runtimes must be created");
  require(SherpaOnnxOrtRuntimeRegisterExecutionProviderLibrary(
              first, registration, library),
          "first plugin registration must succeed");
  require(SherpaOnnxOrtRuntimeRegisterExecutionProviderLibrary(
              second, registration, library),
          "same-name same-path registration must be idempotent");
  require(!SherpaOnnxOrtRuntimeRegisterExecutionProviderLibrary(
              second, registration, conflicting_path),
          "same-name different-path registration must fail");
  require_error_contains("different library path");
  require(SherpaOnnxOrtRuntimeHasExecutionProviderDevice(first, ep, device),
          "registered provider device must be visible");

  SherpaOnnxKeywordSpotterConfig config;
  memset(&config, 0, sizeof(config));
  config.model_config.transducer.encoder = argv[6];
  config.model_config.transducer.decoder = argv[7];
  config.model_config.transducer.joiner = argv[8];
  config.model_config.tokens = argv[9];
  config.model_config.provider = provider_config;
  config.model_config.num_threads = 1;
  config.keywords_file = argv[10];
  config.max_active_paths = 4;
  config.keywords_score = 1.0f;
  config.keywords_threshold = 0.25f;
  const SherpaOnnxKeywordSpotter *spotter =
      SherpaOnnxCreateKeywordSpotter(&config);
  require(spotter != NULL, "accelerated keyword spotter must be created");

  SherpaOnnxDestroyOrtRuntime(first);
  SherpaOnnxDestroyOrtRuntime(second);
  first = SherpaOnnxCreateOrtRuntime();
  require(first != NULL, "spotter-retained runtime must remain acquirable");
  require(SherpaOnnxOrtRuntimeRegisterExecutionProviderLibrary(
              first, registration, library),
          "spotter-retained registration must remain idempotent");
  SherpaOnnxDestroyOrtRuntime(first);

  SherpaOnnxDestroyKeywordSpotter(spotter);
  first = SherpaOnnxCreateOrtRuntime();
  require(first != NULL, "runtime must recreate after spotter teardown");
  require(SherpaOnnxOrtRuntimeRegisterExecutionProviderLibrary(
              first, registration, library),
          "provider must register after complete teardown");
  SherpaOnnxDestroyOrtRuntime(first);
}

static void keyword_exception_contract(int argc, char **argv) {
  if (argc == 1) return;

  SherpaOnnxKeywordSpotterConfig config;
  memset(&config, 0, sizeof(config));
  config.model_config.transducer.encoder = argv[6];
  config.model_config.transducer.decoder = argv[7];
  config.model_config.transducer.joiner = argv[8];
  config.model_config.tokens = argv[9];
  config.model_config.provider = argv[5];
  config.model_config.num_threads = 1;
  config.keywords_file = argv[10];
  config.max_active_paths = 4;
  config.keywords_score = 1.0f;
  config.keywords_threshold = 0.25f;

  require(SherpaOnnxCreateKeywordSpotter(&config) == NULL,
          "accelerated keyword spotter without a retained runtime must fail safely");
}

int main(int argc, char **argv) {
  baseline_contract();
  error_isolation_contract();
  plugin_contract(argc, argv);
  keyword_exception_contract(argc, argv);
  puts("extended sherpa API runtime contract passed");
  return 0;
}
