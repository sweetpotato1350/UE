# step1.관련 패키지 및 모듈 import
import schedule
import time

# step2.실행할 함수 선언
def message():
    print("스케쥴 실행중...")

# step3.실행 주기 설정
schedule.every(1).monday.at("13:30").do()

# step4.스캐쥴 시작
while True:
    schedule.run_pending()
    time.sleep(1)