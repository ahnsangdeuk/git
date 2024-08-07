import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from groq import Groq
from keybert import KeyBERT
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__, static_url_path='/static')

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Groq API 키 설정 (환경 변수에서 가져오기)
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    logger.warning("GROQ_API_KEY is not set in environment variables. Using default key.")
    groq_api_key = "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"

# Groq 클라이언트 초기화
groq_client = Groq(api_key=groq_api_key)

class GroqLLM(LLM):
    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.7

    def _call(self, prompt, stop=None):
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=self.temperature
        )
        return response.choices[0].message.content

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name, "temperature": self.temperature}

    @property
    def _llm_type(self):
        return "groq_llm"

# GroqLLM 인스턴스 생성
llm = GroqLLM()

# KeyBERT 인스턴스 생성
kw_model = KeyBERT()

# 분류 모델 로드 시도
try:
    with open('news_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    model_loaded = True
except FileNotFoundError:
    logger.warning("news_classifier.pkl 파일을 찾을 수 없습니다. 기본 분류 카테고리를 사용합니다.")
    classifier = None
    model_loaded = False

# 한국어 프롬프트 템플릿 설정
summarize_prompt = PromptTemplate(
    input_variables=["text"],
    template="다음 텍스트를 한국어로 간결하게 요약해주세요:\n\n{text}"
)

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="다음 정보를 바탕으로 질문에 한국어로 답변해주세요:\n\n정보: {context}\n\n질문: {question}\n\n답변:"
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/news', methods=['POST'])
def process_news():
    try:
        data = request.json
        url = data.get('url', '')
        question = data.get('question', '')

        logger.info(f"Processing request for URL: {url}, Question: {question}")

        if not url or not question:
            return jsonify({"error": "URL과 질문을 모두 입력해주세요."}), 400

        # URL 검증 및 스킴 추가
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        # 웹 페이지 크롤링
        logger.debug(f"Starting to crawl the webpage: {url}")
        loader = WebBaseLoader(url)
        documents = loader.load()
        logger.debug(f"Crawled {len(documents)} documents")

        # 텍스트 분할
        logger.debug("Starting to split the documents into chunks")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        logger.debug(f"Split documents into {len(texts)} chunks")

        # 텍스트 요약
        logger.debug("Starting text summarization")
        summarize_chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=summarize_prompt)
        summary = summarize_chain.run(texts)
        logger.debug(f"Generated summary: {summary}")

        # 질문 답변
        logger.debug("Starting question answering")
        qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=qa_prompt)
        answer = qa_chain.run(input_documents=texts, question=question)
        logger.debug(f"Generated answer: {answer}")

        # 키워드 추출
        logger.debug("Starting keyword extraction")
        keywords = kw_model.extract_keywords(summary, keyphrase_ngram_range=(1, 2), stop_words=None)
        logger.debug(f"Extracted keywords: {keywords}")

        # 감정 분석
        logger.debug("Starting sentiment analysis")
        sentiment = TextBlob(summary).sentiment
        sentiment_label = "긍정" if sentiment.polarity > 0 else "부정" if sentiment.polarity < 0 else "중립"
        logger.debug(f"Sentiment analysis result: {sentiment_label}")

        # 기사 분류
        category = "분류 불가"
        if model_loaded:
            logger.debug("Starting news classification")
            tfidf_vectorizer = TfidfVectorizer()
            X = tfidf_vectorizer.fit_transform([summary])
            prediction = classifier.predict(X)
            category = prediction[0]
            logger.debug(f"News classification result: {category}")

        return jsonify({
            "summary": summary,
            "answer": answer,
            "keywords": [keyword[0] for keyword in keywords],
            "sentiment": sentiment_label,
            "category": category
        })
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        return jsonify({"error": f"서버 내부 오류가 발생했습니다: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
