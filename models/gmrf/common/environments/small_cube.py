

if __name__ == "__main__":
	import sys
	sys.path.insert(0, "../../..")
	from gdm.common.environment import ObstacleMap

	size = (5, 5, 5)
	sc = ObstacleMap(dimensions=3, size=size, resolution=1)

	## Fill outer walls
	for x in range(0, size[0]):
		for y in range(0, size[1]):
			for z in (0, size[2]-1):
				sc.setCell((x, y, z), 1)
	for y in range(0, size[1]):
		for z in range(0, size[2]):
			for x in (0, size[0]-1):
				sc.setCell((x, y, z), 1)
	for x in range(0, size[0]):
		for z in range(0, size[2]):
			for y in (0, size[1]-1):
				sc.setCell((x, y, z), 1)


	#sc.plot()
	#sc.plot(z=0, vmax=1)
	#sc.plot(z=1, vmax=1)

