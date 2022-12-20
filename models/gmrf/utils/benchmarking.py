import time

class ExecutionTimer:

	def __init__(self, name=""):
		self.__name = name
		self.__timer = time.time()
		#print("["+str(self.__name)+"] measuring elapsed time")
		return

	def getElapsed(self):
		stop_time = time.time()
		elapsed = stop_time - self.__timer
		print("["+str(self.__name)+"] elapsed for "+str(elapsed)+" seconds")
		return elapsed



if __name__ == "__main__":

	t = ExecutionTimer("test")

	dummy = 0
	for i in range(0,10000):
		dummy += i**10

	t.getElapsed()