# PDF 번역기

PDF 파일을 업로드하면 pdf.js로 내용을 보여주고, Upstage Document Parser와 LLM을 이용하여 영어 내용을 한글로 번역해주는 웹 애플리케이션입니다.

## 주요 기능

- PDF 파일 업로드 및 미리보기
- PDF 내용 추출 (Upstage Document Parser 이용)
- 영어 → 한글 번역 (Upstage LLM API 이용)
- 페이지 넘기기 기능

## 설치 방법

1. 저장소 클론
```
git clone <repository-url>
cd pdf-translator
```

2. 필요 패키지 설치
```
pip install -r requirements.txt
```

3. 환경 변수 설정
`.env.example` 파일을 `.env`로 복사한 후 실제 API 키 값으로 수정하세요.
```
cp .env.example .env
```

## 사용 방법

1. 서버 실행
```
python app.py
```

2. 웹 브라우저에서 접속
```
http://localhost:5000
```

3. PDF 파일 업로드 후 번역 결과 확인

## 필요 사항

- Upstage API 키 (Document Parser 및 LLM)
- 파이썬 3.7 이상
- 모던 웹 브라우저 (Chrome, Firefox, Edge 등) 