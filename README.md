# <재활용 안내를 위한 Google Cloud Text-to-Speech 프로젝트>


이 프로젝트는 Google Cloud Text-to-Speech API를 사용하여 감지된 쓰레기 분류에 따라 재활용 안내 음성을 생성합니다.


# 사전 준비 사항
Google Cloud 계정 (Text-to-Speech API 활성화 필요)
Google Cloud 서비스 계정 키 파일 (JSON 형식)
필요한 패키지가 설치된 Python 환경




# <설치 방법>

# 1. 레포지토리 클론 (또는 단일 파일의 경우 스크립트 다운로드):
git clone <repository-url>
cd <repository-directory>

# 2. 필요한 Python 패키지 설치: 
pip install google-cloud-texttospeech




# <사용 방법>

# 1. Google Cloud 서비스 계정 키 파일 업로드:
아래의 코드를 사용하여 키 파일을 업로드합니다.

from google.colab import files

# 파일 업로드
uploaded = files.upload()

# 업로드한 파일 이름을 확인합니다.
for filename in uploaded.keys():
    print(f'Uploaded file: {filename}')

# 2. 환경 변수 설정:

# 업로드한 키 파일의 이름을 환경 변수로 설정합니다.

import os

# 업로드한 파일 이름을 경로로 설정합니다.
key_file = list(uploaded.keys())[0]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f"/content/{key_file}"

# 3. Google Cloud Text-to-Speech 클라이언트 생성:

from google.cloud import texttospeech

# 클라이언트 생성
client = texttospeech.TextToSpeechClient()

# 4. 텍스트 생성 함수 정의:

감지된 쓰레기 클래스 번호에 따라 안내 텍스트를 생성하는 함수를 정의합니다.

def generate_tts_text(class_number):
    if class_number == 1:
        return "이것은 플라스틱 쓰레기입니다. 재활용 방법은 다음과 같습니다."
    elif class_number == 2:
        return "이것은 종이 쓰레기입니다. 재활용 방법은 다음과 같습니다."
    elif class_number == 3:
        return "이것은 금속 캔입니다. 재활용 방법은 다음과 같습니다."
    elif class_number == 4:
        return "이것은 유리병입니다. 재활용 방법은 다음과 같습니다."
    elif class_number == 5:
        return "이것은 페트병입니다. 재활용 방법은 다음과 같습니다."
    elif class_number == 6:
        return "이것은 스티로폼입니다. 재활용 방법은 다음과 같습니다."
    elif class_number == 7:
        return "이것은 비닐입니다. 재활용 방법은 다음과 같습니다."
    elif class_number == 8:
        return "이것은 형광등입니다. 재활용 방법은 다음과 같습니다."
    elif class_number == 9:
        return "이것은 건전지입니다. 재활용 방법은 다음과 같습니다."
    else:
        return "알 수 없는 쓰레기입니다."

# 5. 클래스 번호 설정 및 TTS 요청 생성:

# 클래스 번호 설정
class_number = 1  # 예시로 클래스 번호를 1로 설정합니다. 실제 객체 인식 결과에 따라 이 값을 바꿔야 합니다.

# TTS 요청 생성
text_to_speak = generate_tts_text(class_number)
synthesis_input = texttospeech.SynthesisInput(text=text_to_speak)

# TTS 요청 설정
voice = texttospeech.VoiceSelectionParams(
    language_code="ko-KR",  # 한국어 코드
    ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
)
audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

# 요청 보내기 및 응답 받기
response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

# 6. 응답 저장 및 오디오 파일 다운로드:

# 응답 저장
output_path = "/content/output.mp3"
with open(output_path, "wb") as out:
    out.write(response.audio_content)
    print(f'Audio content written to file "{output_path}"')

# 생성된 오디오 파일 다운로드
from google.colab import files
files.download(output_path)

이제 이 단계를 따라 Google Cloud Text-to-Speech API를 사용하여 재활용 안내 음성을 생성할 수 있습니다.






<참고>
Google Cloud Text-to-Speech 문서: Google Cloud Text-to-Speech Documentation