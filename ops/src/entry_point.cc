
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_lite_custom_op.h"


// Add the forward declaration of RegisterOps from CustomOps namespace
namespace CustomOps {
    void RegisterOps(Ort::CustomOpDomain& domain);
}


extern "C" {
    OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
      fprintf(stderr, "RegisterCustomOps called\n");
      
      OrtStatus* status = nullptr;
      const OrtApi* ort_api = api->GetApi(ORT_API_VERSION);
      
      OrtCustomOpDomain* domain = nullptr;
      status = ort_api->CreateCustomOpDomain("custom", &domain);
      if (status != nullptr) {
        fprintf(stderr, "Failed to create domain\n");
        return status;
      }
    
      Ort::CustomOpDomain custom_domain{domain};
      
      fprintf(stderr, "Calling RegisterOps\n");
      CustomOps::RegisterOps(custom_domain);
      fprintf(stderr, "RegisterOps completed\n");
      
      status = ort_api->AddCustomOpDomain(options, domain);
      fprintf(stderr, "RegisterCustomOps completed\n");
      return status;
    }
}