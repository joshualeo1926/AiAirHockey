import os
import concurrent.futures

def thread(i):
	os.system("call python \"C:\\Users\\josh\\Desktop\\AirHockeySim\\main.py\" " + str(i))

if __name__ == "__main__":
	num_of_games = 4
	with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
		for i in range(num_of_games):
			executor.submit(thread, i)