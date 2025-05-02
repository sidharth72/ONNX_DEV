#include "custom_op_library.h"

#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <vector>
#include <cmath>
#include <mutex>
#include <system_error>


#include "custom_ops.h"
#include "onnxruntime_lite_custom_op.h"

static const char* c_OpDomain = "ai.onnx.custom";


// Create a container to store the custom op library
static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain&& domain) {
    static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
    static std::mutex ort_custom_op_domain_mutex;
    std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
    ort_custom_op_domain_container.push_back(std::move(domain));
}


// Register the custom op library
OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api){
    Ort::Global<void>::api_ = api->GetApi(ORT_API_VERSION);
    OrtStatus* result = nullptr;

    try {
        Ort::CustomOpDomain domain{c_OpDomain};
        CustomCUDAOps::RegisterOps(domain); // Version 1.0

        // Multiple versions can be added to the same domain
        // domain.Add(CustomOps::CustomMatMulV2());
        // domain.Add(CustomOps::CustomMatMulV3()); etc

        Ort::UnownedSessionOptions session_options(options);
        session_options.Add(domain);
        AddOrtCustomOpDomainToContainer(std::move(domain));
    }

    catch(const std::exception& e) {
        Ort::Status status{e};
        result = status.release();
    }

    return result;
}

OrtStatus* ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions* options, const OrtApiBase* api) {
    return RegisterCustomOps(options, api);
  }