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
