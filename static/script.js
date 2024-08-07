document.addEventListener('DOMContentLoaded', () => {
    const urlInput = document.getElementById('url-input');
    const questionInput = document.getElementById('question-input');
    const submitBtn = document.getElementById('submit-btn');
    const loading = document.getElementById('loading');
    const loadingStatus = document.getElementById('loading-status');
    const responseContainer = document.getElementById('response-container');
    const summaryText = document.getElementById('summary-text');
    const answerText = document.getElementById('answer-text');
    const keywordsText = document.getElementById('keywords-text');
    const sentimentText = document.getElementById('sentiment-text');
    const categoryText = document.getElementById('category-text');

    submitBtn.addEventListener('click', async () => {
        const url = urlInput.value.trim();
        const question = questionInput.value.trim();
        if (!url || !question) {
            alert('URL과 질문을 모두 입력해주세요.');
            return;
        }

        loading.classList.remove('hidden');
        responseContainer.classList.add('hidden');
        loadingStatus.textContent = '웹 페이지 크롤링 중...';

        try {
            const response = await fetch('/api/news', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url, question }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'API 요청에 실패했습니다.');
            }

            loadingStatus.textContent = '응답 처리 중...';
            summaryText.textContent = data.summary;
            answerText.textContent = data.answer;
            keywordsText.textContent = data.keywords.join(', ');
            sentimentText.textContent = data.sentiment;
            categoryText.textContent = data.category;
            responseContainer.classList.remove('hidden');
        } catch (error) {
            console.error('Error:', error);
            answerText.textContent = `오류가 발생했습니다: ${error.message}`;
            responseContainer.classList.remove('hidden');
        } finally {
            loading.classList.add('hidden');
        }
    });
});
