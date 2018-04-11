jint throwException(JNIEnv *env, std::string message) {
  jclass exClass;
  std::string className = "java/lang/RuntimeException" ;

  exClass = env->FindClass(className.c_str());

  return env->ThrowNew(exClass, message.c_str());
}

extern "C"
JNIEXPORT void
JNICALL
Java_com_example_ModelWrapper_readFile(
    JNIEnv *env,
    jobject /* this */,
    jstring params_file,
    jint preference) {
  std::string filename = std::string(env->GetStringUTFChars(params_file,
      nullptr));

  if (!nnc::OpenTrainingData(filename.c_str())) {
    throwException("Error on open file: " + filename);
    return;
  }

  if (!nnc::CreateModel()) {
    throwException("Error on create nnapi model");
    return;
  }

  if (!nnc::Compile(preference)) {
    throwException("Error on compile nnapi model");
    return;
  }

  if (!nnc::BuildModel()) {
    throwException("Error on build model");
    return;
  }
}

extern "C"
JNIEXPORT void
JNICALL
Java_com_example_ModelWrapper_cleanup(
    JNIEnv *env,
    jobject /* this */) {
  nnc::Cleanup();
}

extern "C"
JNIEXPORT void
JNICALL
Java_com_example_ModelWrapper_execute(
    JNIEnv *env,
    jobject /* this */) {
  if (!nnc::Execute()) {
    throwException("Error on execute model");
    return;
  }
}

extern "C"
JNIEXPORT void
JNICALL
Java_com_example_ModelWrapper_setInput(
    JNIEnv *env,
    jobject /* this */,
    jbyteArray input_data) {
  jsize input_len = env->GetArrayLength(input_data);

  if (input_len != @TOTAL_INPUT_SIZE) {
    throwException("Input data has wrong length");
    return;
  }

  jbyte *bytes = env->GetByteArrayElements(input_data, 0);

  if (bytes == NULL) {
    throwException("Error on elements from java array input data");
    return;
  }

  if (!nnc::SetInput(bytes)) {
    env->ReleaseByteArrayElements(input_data, bytes, JNI_ABORT);
    throwException("Error on execute model");
    return;
  }

  env->ReleaseByteArrayElements(input_data, bytes, 0);
}

extern "C"
JNIEXPORT jbyteArray
JNICALL
Java_com_example_ModelWrapper_getOutput(
    JNIEnv *env,
    jobject /* this */) {
  jbyteArray result;
  result = env->NewByteArray(@TOTAL_OUTPUT_SIZE);
  if (result == NULL) {
    throwException("out of memory");
    return NULL; /* out of memory error thrown */
  }

  jbyte data[@TOTAL_OUTPUT_SIZE];
  if (!nnc::SetOutput(data)) {
    throwException("Error on execute model");
    return;
  }

  (*env)->SetIntArrayRegion(result, 0, @TOTAL_OUTPUT_SIZE, data);
  return result;
}
