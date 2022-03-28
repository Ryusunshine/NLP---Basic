#한국어 형태소 분석기 MeCab 사용해보기

#테스트 하기
text1 = '루피 아니야" 우리 아이 말이에요 ㅠㅠ 루피 좋아하는 우리 아이, 항상 껴안고 다녀 많이 헤졌어요'
text2 = " 화이트로 구매했는데...왜 다른제품으로 오죠? 외국에서 배송받지않았다면...당장 반품인데~~~"
text3 = "아주아주 마음에 쏙쏙 듭니다.판매자님 대응 마음에 들어요.앞으로도 자주 구매원합니다.번창하세요.고맙습니다.복받으실거에요"

#MeCab 불러오기
import MeCab
m = MeCab.Tagger()
print(m.parse(text3))

data = m.parse("오늘 저녁으로 돈까스를 먹을까 써브웨이를 먹을까 너무 어려운 고민이야")
data = data.replace("EOS", " ") #모르는 오류 삭제
data = data.rstrip() # 우측 공백 삭제
print(data)

word = []
for i in data.split("\n"): #줄바꿈으로 생성된 리스트 출력하기
    data = i.split(',')[0].split(' ')
    word.append(data)

print(word)
print(word[0][0], word[0],[1])
print(word[1][0], word[1],[1])

#데이터프레임 형식으로 출력하기
import pandas as pd
column_name = ['형태소', '품사']
df = pd.DataFrame(word, columns = column_name)
print(df)

#텍스트 분석 연습하기

#파일읽기
with open("C:\\Users\\ryusu\\바탕 화면\\File\\text.txt", encoding='utf-8') as f:
    text = f.read()
    print(text)

#데이터 전처리(1)후 데이터프레임으로 출력
data=m.parse(text)
data = data.replace("EOS", " ")
data = data.rstrip()

word = []
for i in data.split("\n"):
    data = i.split(',')[0].split(' ')
    word.append(data)

print(word)

#빈도 분석을 위한 전처리
text_filter = str(word)
text_filter= text_filter.replace('(',' ')
text_filter= text_filter.replace(')',' ')
text_filter= text_filter.replace('+',' ')
text_filter= text_filter.replace("'",' ')
text_filter= text_filter.replace('.',' ')
print(text_filter)

#형태소 분석 결과를 저장한 파일을 생성한다

f = open("C:\\Users\\ryusu\\바탕 화면\\File\\test result.txt", 'w')
f.write(text_filter)
f.close