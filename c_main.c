#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include "onnxruntime-linux-x64-gpu-1.24.4/include/onnxruntime_c_api.h"

#define CHK(x) do { OrtStatus *_s = (x); if (_s) { fprintf(stderr, "%s\n", api->GetErrorMessage(_s)); api->ReleaseStatus(_s); exit(1); } } while(0)

int main(void) {
	if (!dlopen(".venv/lib/python3.10/site-packages/torch/lib/libtorch.so", RTLD_LAZY | RTLD_GLOBAL)) {
		fprintf(stderr, "libtorch: %s\n", dlerror());
		return 1;
	}
	void *h = dlopen("onnxruntime-linux-x64-gpu-1.24.4/lib/libonnxruntime.so.1.24.4", RTLD_LAZY | RTLD_GLOBAL);
	const OrtApi *api = ((const OrtApiBase *(*)(void))dlsym(h, "OrtGetApiBase"))()->GetApi(ORT_API_VERSION);

	OrtEnv *env;
	CHK(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "repro", &env));

	OrtSessionOptions *opts;
	CHK(api->CreateSessionOptions(&opts));

	OrtStatus *(*AppendCUDA)(OrtSessionOptions *, int) = dlsym(h, "OrtSessionOptionsAppendExecutionProvider_CUDA");
	CHK(AppendCUDA(opts, 0));

	OrtSession *sess;
	CHK(api->CreateSession(env, "decoder_joint-model.onnx", opts, &sess));

	OrtMemoryInfo *mem;
	CHK(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem));

	float enc[1024] = {0};
	int64_t enc_sh[] = {1, 1024, 1};
	OrtValue *v_enc;
	CHK(api->CreateTensorWithDataAsOrtValue(mem, enc, sizeof(enc), enc_sh, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v_enc));

	int32_t tgt[1] = {0};
	int64_t tgt_sh[] = {1, 1};
	OrtValue *v_tgt;
	CHK(api->CreateTensorWithDataAsOrtValue(mem, tgt, sizeof(tgt), tgt_sh, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &v_tgt));

	int32_t tlen[1] = {1};
	int64_t tlen_sh[] = {1};
	OrtValue *v_tlen;
	CHK(api->CreateTensorWithDataAsOrtValue(mem, tlen, sizeof(tlen), tlen_sh, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &v_tlen));

	float s1[2*640] = {0};
	int64_t s1_sh[] = {2, 1, 640};
	OrtValue *v_s1;
	CHK(api->CreateTensorWithDataAsOrtValue(mem, s1, sizeof(s1), s1_sh, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v_s1));

	float s2[2*640] = {0};
	int64_t s2_sh[] = {2, 1, 640};
	OrtValue *v_s2;
	CHK(api->CreateTensorWithDataAsOrtValue(mem, s2, sizeof(s2), s2_sh, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v_s2));

	const char *in_names[] = {"encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"};
	const OrtValue *ins[] = {v_enc, v_tgt, v_tlen, v_s1, v_s2};

	size_t n_out;
	CHK(api->SessionGetOutputCount(sess, &n_out));
	OrtAllocator *alloc;
	CHK(api->GetAllocatorWithDefaultOptions(&alloc));

	const char **out_names = malloc(n_out * sizeof(char *));
	OrtValue **outs = calloc(n_out, sizeof(OrtValue *));
	for (size_t i = 0; i < n_out; i++) {
		char *name;
		CHK(api->SessionGetOutputName(sess, i, alloc, &name));
		out_names[i] = name;
	}

	CHK(api->Run(sess, NULL, in_names, ins, 5, out_names, n_out, outs));
	printf("done\n");

	for (size_t i = 0; i < n_out; i++) { api->ReleaseValue(outs[i]); alloc->Free(alloc, (void *)out_names[i]); }
	free(outs); free(out_names);
	api->ReleaseValue(v_enc); api->ReleaseValue(v_tgt); api->ReleaseValue(v_tlen);
	api->ReleaseValue(v_s1); api->ReleaseValue(v_s2);
	api->ReleaseMemoryInfo(mem);
	api->ReleaseSession(sess);
	api->ReleaseSessionOptions(opts);
	api->ReleaseEnv(env);

	dlclose(h);
}
