"#include <sys/types.h>\n\
#include <sys/mman.h>\n\
#include <sys/stat.h>\n\
#include <unistd.h>\n\
#include <fcntl.h>\n\
#include <android/log.h>\n\
#include <android/NeuralNetworks.h>\n\
#include <string>\n\
\n\
#include \"nn.h\"\n\
\n\
#define LOG_TAG \"NNC\"\n\
\n\
namespace nnc {\n\
\n\
static ANeuralNetworksMemory* mem = NULL;\n\
static int fd;\n\
static ANeuralNetworksModel* model = NULL;\n\
static ANeuralNetworksCompilation* compilation = NULL;\n\
static ANeuralNetworksExecution* run = NULL;\n\
\n\
bool OpenTrainingData(const char* file_name) {\n\
  int fd = open(file_name, O_RDONLY);\n\
\n\
  if (fd < 0) {\n\
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,\n\
                        \"open failed\");\n\
    return false;\n\
  }\n\
\n\
  struct stat sb;\n\
  fstat(fd, &sb);\n\
  size_t buffer_size_bytes = sb.st_size;\n\
  int status = ANeuralNetworksMemory_createFromFd(buffer_size_bytes, PROT_READ, fd, 0, &mem);\n\
  if (status != ANEURALNETWORKS_NO_ERROR) {\n\
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,\n\
                        \"ANeuralNetworksMemory_createFromFd failed\");\n\
    return false;\n\
  }\n\
\n\
  return true;\n\
}\n\
\n\
bool CreateModel() {\n\
  int status = ANeuralNetworksModel_create(&model);\n\
\n\
  if (status != ANEURALNETWORKS_NO_ERROR) {\n\
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,\n\
                        \"ANeuralNetworksMemory_createFromFd failed\");\n\
    return false;\n\
  }\n\
\n\
  return true;\n\
}\n\
\n\
bool Compile(int32_t preference) {\n\
  int status = ANeuralNetworksCompilation_create(model, &compilation);\n\
  if (status != ANEURALNETWORKS_NO_ERROR) {\n\
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,\n\
                        \"ANeuralNetworksMemory_createFromFd failed\");\n\
    return false;\n\
  }\n\
\n\
  status = ANeuralNetworksCompilation_setPreference(compilation, preference);\n\
  if (status != ANEURALNETWORKS_NO_ERROR) {\n\
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,\n\
                        \"ANeuralNetworksMemory_createFromFd failed\");\n\
    return false;\n\
  }\n\
\n\
  return true;\n\
}\n\
\n\
\n\
bool Execute() {\n\
  int status = ANeuralNetworksExecution_create(compilation, &run);\n\
\n\
    if (status != ANEURALNETWORKS_NO_ERROR) {\n\
      __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,\n\
                          \"ANeuralNetworksMemory_createFromFd failed\");\n\
      return false;\n\
    }\n\
\n\
  ANeuralNetworksEvent* run_end = NULL;\n\
  ANeuralNetworksExecution_startCompute(run, &run_end);\n\
  ANeuralNetworksEvent_wait(run_end);\n\
  ANeuralNetworksEvent_free(run_end);\n\
  ANeuralNetworksExecution_free(run);\n\
  return true;\n\
}\n\
\n\
void Cleanup() {\n\
  ANeuralNetworksCompilation_free(compilation);\n\
  ANeuralNetworksModel_free(model);\n\
  ANeuralNetworksMemory_free(mem);\n\
}\n\
\n\
#define CHECK_ADD_SCALAR(x)                           \\\n\
  if (!x) {                                           \\\n\
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,   \\\n\
        \"AddScalar Failed\");                        \\\n\
    return false;                                     \\\n\
  }\n\
\n\
bool AddScalarInt32(int32_t id, int value) {\n\
  ANeuralNetworksOperandType operand_type{.type = ANEURALNETWORKS_INT32};\n\
\n\
  int status =  ANeuralNetworksModel_addOperand(model, &operand_type);\n\
  if (status != ANEURALNETWORKS_NO_ERROR) {\n\
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,\n\
        \"ANeuralNetworksModel_addOperand failed\");\n\
    return false;\n\
  }\n\
\n\
  status = ANeuralNetworksModel_setOperandValue(model, id, &value, sizeof(int32_t));\n\
  if (status != ANEURALNETWORKS_NO_ERROR) {\n\
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,\n\
        \"ANeuralNetworksModel_setOperandValue failed\");\n\
    return false;\n\
  }\n\
\n\
  return true;\n\
}\n\
\n\
bool AddScalarFloat32(int32_t id, float value) {\n\
  ANeuralNetworksOperandType operand_type{.type = ANEURALNETWORKS_FLOAT32};\n\
\n\
  int status =  ANeuralNetworksModel_addOperand(model, &operand_type);\n\
  if (status != ANEURALNETWORKS_NO_ERROR) {\n\
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,\n\
        \"ANeuralNetworksModel_addOperand failed\");\n\
    return false;\n\
  }\n\
\n\
  status = ANeuralNetworksModel_setOperandValue(model, id, &value, sizeof(float));\n\
  if (status != ANEURALNETWORKS_NO_ERROR) {\n\
    __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,\n\
        \"ANeuralNetworksModel_setOperandValue failed\");\n\
    return false;\n\
  }\n\
\n\
  return true;\n\
}\n\
\n\
bool BuildModel() {\n\
  int tensor_size = 0;\n\
  int offset = 0;\n\
  int status;\n\
 ";
