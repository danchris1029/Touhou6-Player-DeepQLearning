#	Christian G.
#	Description:
#	Project to experiement with a Q-Learning model to see if I could make a bot that can play Touhou 6: Embodiment of Scarlet Devil.
#	Took about 12 hours to put together and experiement to get some decent output.

#	Note: Bot couldn't make it past the first level, maybe nudging some variables could ecourage better performance.

import ctypes, win32ui, win32process, win32api, win32con, win32gui
import numpy as np
import time
import Input 
from PIL import ImageGrab
import PIL
from simple_dqn import Agent
import matplotlib.pyplot as plt
from os import path


LONG = ctypes.c_long
DWORD = ctypes.c_ulong
ULONG_PTR = ctypes.POINTER(DWORD)
WORD = ctypes.c_ushort

PROCESS_ALL_ACCESS = 0x1F0FFF
hw = win32ui.FindWindow(None, "Touhou Scarlet Devil Land ~ The Embodiment of Scarlet Devil v1.02h").GetSafeHwnd() # Rename to the window
PID = win32process.GetWindowThreadProcessId(hw)[1]
PROCESS = win32api.OpenProcess(PROCESS_ALL_ACCESS,0,PID)		# Requires running as administrator
rPM = ctypes.windll.kernel32.ReadProcessMemory

def getScore(PROCESS):
	scoreAddress = 0x0069BCA4
	score = ctypes.create_string_buffer(4)
	rPM(PROCESS.handle, scoreAddress, score, 4, 0)		   # Score is 4 bytes
	s = int.from_bytes(score.value, "little")
	return s

def getLives(PROCESS):
	livesAddress = 0x0069D4BA
	lives = ctypes.create_string_buffer(1)
	rPM(PROCESS.handle, livesAddress, lives, 1, 0)			# Lives is one byte
	l = int.from_bytes(lives.value, "little")
	return l												# release memory



def take_screenshot():
	hwnd = win32gui.GetForegroundWindow()
	bbox = win32gui.GetWindowRect(hwnd)
	img = ImageGrab.grab(bbox)
	img = img.resize((25, 25))								# Edit dimensions if not enough memory to allocate
	img = np.asarray(img)
	img = img[:, :, 0]										# Convert RBG image to greyscale for less memory usage but still visible
	return np.ndarray.flatten(img)

def takeAction(action):									 # Ai control is slow, could be implemented better to allow more efficient flexibility
	# Hidden no-action when 0
	if action == 0:
		print("NONE")
	if action == 1:
		print("Z")
		Input.PressKey(Input.KEY_Z)
		time.sleep(0.2)
		Input.ReleaseKey(Input.KEY_Z)
	if action == 2:
		print("LEFT")
		Input.PressKey(Input.KEY_LEFT)
		time.sleep(0.2)
		Input.ReleaseKey(Input.KEY_LEFT)
	if action == 3:
		print("RIGHT")
		Input.PressKey(Input.KEY_RIGHT)
		time.sleep(0.2)
		Input.ReleaseKey(Input.KEY_RIGHT)
	if action == 4:
		print("UP")
		Input.PressKey(Input.KEY_UP)
		time.sleep(0.2)
		Input.ReleaseKey(Input.KEY_UP)
	if action == 5:
		print("DOWN")
		Input.PressKey(Input.KEY_DOWN)
		time.sleep(0.2)
		Input.ReleaseKey(Input.KEY_DOWN)
	if action == 6:
		print("SHIFT")
		Input.PressKey(Input.KEY_SHIFT)
		time.sleep(0.2)
		Input.ReleaseKey(Input.KEY_SHIFT)
		
def exitPlay():												# If all lives are loss bot returns to main menu
	Input.PressKey(Input.KEY_ESC)
	print("press")
	time.sleep(0.2)
	Input.ReleaseKey(Input.KEY_ESC)
	time.sleep(0.5)
	
	Input.PressKey(Input.KEY_DOWN)
	print("press")
	time.sleep(0.2)
	Input.ReleaseKey(Input.KEY_DOWN)
	time.sleep(0.5) 
	
	Input.PressKey(Input.KEY_Z)
	print("press")
	time.sleep(0.2)
	Input.ReleaseKey(Input.KEY_Z)
	time.sleep(0.5)	 
	
	Input.PressKey(Input.KEY_UP)
	print("press")
	time.sleep(0.2)
	Input.ReleaseKey(Input.KEY_UP)
	time.sleep(0.5)	 

	Input.PressKey(Input.KEY_Z)
	print("press")
	time.sleep(0.2)
	Input.ReleaseKey(Input.KEY_Z)
	time.sleep(0.5)	 
	
def startFromMenu():								# bot goes through menu to start the game
	Input.PressKey(Input.KEY_Z)
	print("press")
	time.sleep(0.2)
	Input.ReleaseKey(Input.KEY_Z)
	time.sleep(0.5)
	Input.PressKey(Input.KEY_Z)
	print("press")
	time.sleep(0.2)
	Input.ReleaseKey(Input.KEY_Z)
	time.sleep(1.5)
	Input.PressKey(Input.KEY_Z)
	print("press")
	time.sleep(0.2)
	Input.ReleaseKey(Input.KEY_Z)
	time.sleep(0.5)
	Input.PressKey(Input.KEY_Z)
	print("press")
	time.sleep(0.2)
	Input.ReleaseKey(Input.KEY_Z)
	time.sleep(0.5)
	Input.PressKey(Input.KEY_Z)
	print("press")
	time.sleep(0.2)
	Input.ReleaseKey(Input.KEY_Z)
	time.sleep(1)
	

def restart():												
	exitPlay()
	time.sleep(2)
	startFromMenu()
	time.sleep(1)
	
n_games = 50
lr = 0.005
reward = 0
max_reward = 0
times_maxed = 0
agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, input_dims=[625],			# Yikes, takes alot of memory (Atleast for me).
			   n_actions=7, mem_size=1000000, batch_size=64)
			   
if path.exists("./checkpoint/2hu.ckpt.meta"):							# Does the checkpoint exist? Then load checkpoint
	print("loading model")
	agent.load_models()			   

print("")
print("Be on the starting screen (not the main menu where 'start' is located), Click the game's window and don't touch! pretty plz :)")
print("Starting in...")
for i in range(3, 0, -1):		# Give 3 seconds for to focus on the game's window
	print(i)
	time.sleep(1)
print("GO!")


startFromMenu()
for i in range(n_games):				
	done = False
	lives = getLives(PROCESS)
	prev_lives = lives
	reward = 0
	time.sleep(1)
	agent.epsilon = 1.0
	observation = take_screenshot()
	print("round ", i)
	prev_t = 0
	prev_reward = 0
	gain = 0
	prev_gain = 0
	no_action_multiplier = 1
	reward_multiplier = 1
	while not done:
		t = time.time()
		action = agent.choose_action(observation)
		takeAction(action)
		observation_ = take_screenshot()
		lives = getLives(PROCESS)
		
		if (t - prev_t > 3):									#TODO: Messy reward system is messy, should be cleaned up.
			prev_t = t						
			reward_multiplier += 1
			reward = gain+prev_reward
			prev_reward = reward
			if prev_gain == gain:
				reward -= no_action_multiplier
				no_action_multiplier += 1
				print("no gain in last 3 sec")
			else:
				reward += reward_multiplier
				no_action_multiplier = 1
			prev_gain = gain
			print("reward_multiplier: ", reward_multiplier)
			print("no_action_multipler: ", no_action_multiplier)
		if lives != prev_lives:
			reward_multiplier = 1
			print("died")								# take away points for dying
		prev_lives = lives
		gain = getScore(PROCESS) - prev_reward
		done = bool(lives == 0)
		agent.store_transition(observation, action, reward, observation_, done)
		agent.learn()
		
	print("saving model")	
	agent.save_models() 
	if reward > max_reward:
		max_reward = reward
		times_maxed += 1
		print("New max reward of ", max_reward)
		print("Times maxed: " , times_maxed)
	else:
		print("Getting dumber")
		print("Times maxed: " , times_maxed)
	if not (i == n_games-1):
		restart()
	
 
print("The 2hu's have learnt :)") 		