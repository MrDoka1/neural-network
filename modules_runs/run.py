from modules_runs.detect_face import start
from modules_runs.load_model import predict_emotion

if __name__ == '__main__':
    start(predict_function=predict_emotion)