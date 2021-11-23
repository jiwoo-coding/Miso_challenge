import pandas as pd
from sklearn import metrics

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


name = 'skin'
modelPath = '~/model_saved/Skin/Skin_DenseNet169_50/model-001-0.432977-0.262467.h5' #'e:/model_saved/entropion/resnet_v1_50_100/model-038-0.958750-0.965174.h5'  # 모델이 저장된 경로

# weight = 'model-078-0.925417-0.916944.h5'        # 학습된 모델의 파일이름
test_Path = '/mnt/hackerton/dataset/Skin/Test/val_verify'#'e:/hackathon/entropion/val' # 테스트 이미지 폴더
#dataset 이후 Skin 혹은 Eye로 데이터셋 변경이 가능하며, 그 이후 디렉토리 구조는 동일합니다.

# model = load_model(modelPath + weight)
model = load_model(modelPath)
datagen_test = ImageDataGenerator(rescale=1./255)

generator_test = datagen_test.flow_from_directory(directory=test_Path,
                                                  target_size=(224, 224),
                                                  batch_size=256,
                                                  shuffle=False)

# model로 test set 추론
generator_test.reset()
cls_test = generator_test.classes
cls_pred = model.predict_generator(generator_test, verbose=1, workers=0)
cls_pred_argmax = cls_pred.argmax(axis=1)

# 결과 산출 및 저장
report = metrics.classification_report(y_true=cls_test, y_pred=cls_pred_argmax, output_dict=True)
report = pd.DataFrame(report).transpose()
#report.to_csv(f'e:/output/report_test_{name}.csv', index=True, encoding='cp949')
report.to_csv(f'~/report_test_{name}.csv', index=True, encoding='cp949')
print(report)