"namespace nnc {\n\
\n\
static ANeuralNetworksMemory* mem = NULL;\n\
static int fd;\n\
static ANeuralNetworksModel* model = NULL;\n\
statuc ANeuralNetworksCompilation* compilation = NULL;\n\
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
  fstat(mmap_fd_, &sb);\n\
  size_t buffer_size_bytes = sb.st_size;\n\
  int status = ANeuralNetworksMemory_createFromFd(buffer_size_bytes, PROT_READ, fd, 0, mem);\n\
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
bool CreateExecutionInstance(ANeuralNetworksExecution** run) {\n\
  int status = ANeuralNetworksExecution_create(compilation, run);\n\
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
bool Execute(ANeuralNetworksExecution* run) {\n\
  ANeuralNetworksEvent* run_end = NULL;\n\
  ANeuralNetworksExecution_startCompute(run, &run_end);\n\
  ANeuralNetworksEvent_wait(run_end);\n\
  ANeuralNetworksEvent_free(run_end);\n\
}\n\
\n\
 bool BuildModel() {\n\
 ";

