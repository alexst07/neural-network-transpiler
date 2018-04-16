"#include <jni.h>\n\
#include <string>\n\
#include \"nn.h\"\n\
\n\
jint throwException(JNIEnv *env, std::string message) {\n\
  jclass exClass;\n\
  std::string className = \"java/lang/RuntimeException\" ;\n\
\n\
  exClass = env->FindClass(className.c_str());\n\
\n\
  return env->ThrowNew(exClass, message.c_str());\n\
}\n\
\n\
extern \"C\"\n\
JNIEXPORT void\n\
JNICALL\n\
@JAVA_PACKAGE_readFile(\n\
    JNIEnv *env,\n\
    jobject /* this */,\n\
    jstring params_file,\n\
    jint preference) {\n\
  std::string filename = std::string(env->GetStringUTFChars(params_file,\n\
      nullptr));\n\
\n\
  if (!nnc::OpenTrainingData(filename.c_str())) {\n\
    throwException(env, \"Error on open file: \" + filename);\n\
    return;\n\
  }\n\
\n\
  if (!nnc::CreateModel()) {\n\
    throwException(env, \"Error on create nnapi model\");\n\
    return;\n\
  }\n\
\n\
  if (!nnc::Compile(preference)) {\n\
    throwException(env, \"Error on compile nnapi model\");\n\
    return;\n\
  }\n\
\n\
  if (!nnc::BuildModel()) {\n\
    throwException(env, \"Error on build model\");\n\
    return;\n\
  }\n\
}\n\
\n\
extern \"C\"\n\
JNIEXPORT void\n\
JNICALL\n\
@JAVA_PACKAGE_cleanup(\n\
    JNIEnv *env,\n\
    jobject /* this */) {\n\
  nnc::Cleanup();\n\
}\n\
\n\
extern \"C\"\n\
JNIEXPORT void\n\
JNICALL\n\
@JAVA_PACKAGE_execute(\n\
    JNIEnv *env,\n\
    jobject /* this */) {\n\
  if (!nnc::Execute()) {\n\
    throwException(env, \"Error on execute model\");\n\
    return;\n\
  }\n\
}\n\
\n\
extern \"C\"\n\
JNIEXPORT void\n\
JNICALL\n\
@JAVA_PACKAGE_setInput(\n\
    JNIEnv *env,\n\
    jobject /* this */,\n\
    jbyteArray input_data) {\n\
  jsize input_len = env->GetArrayLength(input_data);\n\
\n\
  if (input_len != @TOTAL_INPUT_SIZE) {\n\
    throwException(env, \"Input data has wrong length\");\n\
    return;\n\
  }\n\
\n\
  jbyte *bytes = env->GetByteArrayElements(input_data, 0);\n\
\n\
  if (bytes == NULL) {\n\
    throwException(env, \"Error on elements from java array input data\");\n\
    return;\n\
  }\n\
\n\
  if (!nnc::SetInput(bytes)) {\n\
    env->ReleaseByteArrayElements(input_data, bytes, JNI_ABORT);\n\
    throwException(env, \"Error on execute model\");\n\
    return;\n\
  }\n\
\n\
  env->ReleaseByteArrayElements(input_data, bytes, 0);\n\
}\n\
\n\
extern \"C\"\n\
JNIEXPORT jbyteArray\n\
JNICALL\n\
@JAVA_PACKAGE_getOutput(\n\
    JNIEnv *env,\n\
    jobject /* this */) {\n\
  jbyteArray result;\n\
  result = env->NewByteArray(@TOTAL_OUTPUT_SIZE);\n\
  if (result == NULL) {\n\
    throwException(env, \"out of memory\");\n\
    return NULL; /* out of memory error thrown */\n\
  }\n\
\n\
  jbyte data[@TOTAL_OUTPUT_SIZE];\n\
  if (!nnc::SetOutput(data)) {\n\
    throwException(env, \"Error on execute model\");\n\
    return NULL;\n\
  }\n\
\n\
  env->SetByteArrayRegion(result, 0, @TOTAL_OUTPUT_SIZE, data);\n\
  return result;\n\
}\n\
"
