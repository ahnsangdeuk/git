from flask import Flask, render_template, jsonify, request, send_from_directory
import requests
import xmltodict
import json
import logging

app = Flask(__name__, static_folder='static')
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/get_attractions', methods=['GET'])
def get_attractions():
    page_no = request.args.get('pageNo', default='1')
    num_of_rows = request.args.get('numOfRows', default='10')
    keyword = request.args.get('keyword', default='')
    
    app.logger.info(f"Received request with pageNo: {page_no}, numOfRows: {num_of_rows}, keyword: {keyword}")
    
    url = 'http://apis.data.go.kr/6270000/getTourKorAttract/getTourKorAttractList'
    params = {
        'serviceKey': 'Poc6rnzr84pjw40+/XOt70+NL37qgNMjsHeh1V/xVwVU3ioy/BeGDnz1TOjcbwCDnnGPT4Sbn/GVsshKDZ8F0Q==',
        'pageNo': page_no,
        'numOfRows': num_of_rows,
        'type': 'xml'
    }
    
    if keyword:
        params['keyword'] = keyword
    
    try:
        app.logger.info(f"Sending request to API with params: {params}")
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        app.logger.info(f"Received response with status code: {response.status_code}")
        app.logger.debug(f"Response content: {response.text}")
        
        data_dict = xmltodict.parse(response.content)
        app.logger.debug(f"Parsed XML: {json.dumps(data_dict, indent=2)}")
        
        if 'response' not in data_dict:
            app.logger.error("Unexpected API response structure")
            return jsonify({"error": "Unexpected API response structure"}), 500
        
        body = data_dict['response'].get('body', {})
        items = body.get('items', {}).get('item', [])
        total_count = int(body.get('totalCount', 0))
        
        if isinstance(items, dict):
            items = [items]
        
        attractions = []
        for item in items:
            attractions.append({
                'id': item.get('contentid', ''),
                'attractname': item.get('attractname', '이름 없음'),
                'address': item.get('address', '주소 정보 없음'),
                'attractcontents': item.get('attractcontents', '설명 없음'),
                'image': item.get('mainimgthumb', ''),
                'latitude': item.get('mapy', ''),
                'longitude': item.get('mapx', '')
            })
        
        app.logger.info(f"Returning {len(attractions)} attractions, total count: {total_count}")
        return jsonify({
            'attractions': attractions,
            'totalCount': total_count,
            'pageNo': int(page_no),
            'numOfRows': int(num_of_rows)
        })
    
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Request failed: {str(e)}")
        return jsonify({"error": f"Request to external API failed: {str(e)}"}), 500
    except Exception as e:
        app.logger.error(f"Error processing response: {str(e)}")
        return jsonify({"error": f"Error processing API response: {str(e)}"}), 500

@app.route('/attraction/<id>')
def attraction_detail(id):
    # 여기에서 특정 관광지의 상세 정보를 가져오는 API 호출을 구현해야 합니다.
    # 지금은 임시로 더미 데이터를 반환합니다.
    attraction = {
        'id': id,
        'attractname': f'관광지 {id}',
        'address': '대전시 어딘가',
        'attractcontents': '이 관광지에 대한 자세한 설명입니다.',
        'image': 'https://via.placeholder.com/400x300',
        'latitude': '36.3504',
        'longitude': '127.3845',
        'rating': 4.5
    }
    return render_template('detail.html', attraction=attraction)

if __name__ == '__main__':
    app.run(debug=True)