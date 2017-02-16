package zcash

import (
	"bytes"
	"log"
	"math"
	"testing"
	"fmt"
	"unsafe"
	"time"


	"github.com/kilo17/go-opencl/cl"
	"github.com/kilo17/gominer2/clients"
	"github.com/kilo17/gominer2/mining"
)



)

var provenSolutions = []struct {
	height          int
	hash            string
	workHeader      []byte
	offset          int
	submittedHeader []byte
	intensity       int
}{
	{
		height:          56206,
		hash:            "00000000000006418b86014ff54b457f52665b428d5af57e80b0b7ec84c706e5",
		workHeader:      []byte{0, 0, 0, 0, 0, 0, 26, 158, 25, 209, 169, 53, 113, 22, 90, 11, 72, 7, 222, 103, 247, 244, 163, 156, 158, 5, 53, 126, 186, 215, 88, 48, 45, 32, 0, 0, 0, 0, 0, 0, 20, 25, 103, 87, 0, 0, 0, 0, 218, 189, 84, 137, 247, 169, 197, 113, 213, 120, 125, 148, 92, 197, 47, 212, 250, 153, 114, 53, 199, 209, 183, 97, 28, 242, 206, 120, 191, 202, 34, 9},
		offset:          5 * int(math.Exp2(float64(28))),
		submittedHeader: []byte{0, 0, 0, 0, 0, 0, 26, 158, 25, 209, 169, 53, 113, 22, 90, 11, 72, 7, 222, 103, 247, 244, 163, 156, 158, 5, 53, 126, 186, 215, 88, 48, 88, 47, 107, 95, 0, 0, 0, 0, 20, 25, 103, 87, 0, 0, 0, 0, 218, 189, 84, 137, 247, 169, 197, 113, 213, 120, 125, 148, 92, 197, 47, 212, 250, 153, 114, 53, 199, 209, 183, 97, 28, 242, 206, 120, 191, 202, 34, 9},
		intensity:       28,
	},
	{
		height:          57653,
		hash:            "00000000000001ccac64b49a9ebc69c6046a93f4d32d8f8f6967c8f487ed8cec",
		workHeader:      []byte{0, 0, 0, 0, 0, 0, 6, 72, 174, 217, 105, 206, 174, 59, 150, 117, 251, 55, 209, 192, 241, 37, 35, 184, 2, 194, 253, 173, 207, 249, 114, 1, 62, 26, 0, 0, 0, 0, 0, 0, 41, 7, 115, 87, 0, 0, 0, 0, 56, 56, 181, 217, 76, 24, 251, 231, 137, 4, 166, 20, 40, 53, 77, 36, 148, 23, 138, 146, 2, 199, 168, 122, 71, 162, 44, 150, 144, 2, 198, 67},
		offset:          805306368,
		submittedHeader: []byte{0, 0, 0, 0, 0, 0, 6, 72, 174, 217, 105, 206, 174, 59, 150, 117, 251, 55, 209, 192, 241, 37, 35, 184, 2, 194, 253, 173, 207, 249, 114, 1, 7, 235, 26, 63, 0, 0, 0, 0, 41, 7, 115, 87, 0, 0, 0, 0, 56, 56, 181, 217, 76, 24, 251, 231, 137, 4, 166, 20, 40, 53, 77, 36, 148, 23, 138, 146, 2, 199, 168, 122, 71, 162, 44, 150, 144, 2, 198, 67},
		intensity:       28,
	},
}





// Miner actually mines :-)
type Miner1 struct {
	ClDevices       map[int]*cl.Device
	HashRateReports chan *mining.HashRateReport
	Client          clients.Client
}

//singleDeviceMiner actually mines on 1 opencl device
type singleDeviceMiner1 struct {
	ClDevice        *cl.Device
	MinerID         int
	HashRateReports chan *mining.HashRateReport
	Client          clients.Client
}

//Mine spawns a seperate miner for each device defined in the CLDevices and feeds it with work
func (m *Miner) Mine() {

	m.Client.Start()
	for minerID, device := range m.ClDevices {
		sdm := &singleDeviceMiner{
			ClDevice:        device,
			MinerID:         minerID,
			HashRateReports: m.HashRateReports,
			Client:          m.Client,
		}
		go sdm.mine()
	}
}

type solst1 struct {
	nr             uint32
	likelyInvalids uint32
	valid          [maxSolutions]uint8
	values         [maxSolutions][(1 << equihashParamK)]uint32
}

func numberOfComputeUnits1(gpu string) int {
	//if gpu == "rx480" {
	//}
	if gpu == "Fiji" {
		return 56
	}

	log.Panicln("Unknown GPU: ", gpu)
	return 0
}

func selectWorkSizeBlake1() (workSize int) {
	workSize =
		64 * /* thread per wavefront */
			blakeWPS * /* wavefront per simd */
			4 * /* simd per compute unit */
			numberOfComputeUnits("Fiji")
	// Make the work group size a multiple of the nr of wavefronts, while
	// dividing the number of inputs. This results in the worksize being a
	// power of 2.
	for (numberOfInputs % workSize) != 0 {
		workSize += 64
	}
	return
}

func (miner *singleDeviceMiner) mine() {
	rowsPerUint := 4
	if numberOfSlots < 16 {
		rowsPerUint = 8
	}

	log.Println(miner.MinerID, "- Initializing", miner.ClDevice.Type(), "-", miner.ClDevice.Name())

	context, err := cl.CreateContext([]*cl.Device{miner.ClDevice})
	if err != nil {
		log.Fatalln(miner.MinerID, "-", err)
	}
	defer context.Release()

	commandQueue, err := context.CreateCommandQueue(miner.ClDevice, 0)
	if err != nil {
		log.Fatalln(miner.MinerID, "-", err)
	}
	defer commandQueue.Release()

	program, err := context.CreateProgramWithSource([]string{kernelSource})
	if err != nil {
		log.Fatalln(miner.MinerID, "-", err)
	}
	defer program.Release()

	err = program.BuildProgram([]*cl.Device{miner.ClDevice}, "")
	if err != nil {
		log.Fatalln(miner.MinerID, "-", err)
	}

	//Create kernels
	kernelInitHt, err := program.CreateKernel("kernel_init_ht")
	if err != nil {
		log.Fatalln(miner.MinerID, "-", err)
	}
	defer kernelInitHt.Release()

	var kernelRounds [equihashParamK]*cl.Kernel
	for round := 0; round < equihashParamK; round++ {
		kernelRounds[round], err = program.CreateKernel(fmt.Sprintf("kernel_round%d", round))
		if err != nil {
			log.Fatalln(miner.MinerID, "-", err)
		}
		defer kernelRounds[round].Release()
	}

	kernelSolutions, err := program.CreateKernel("kernel_sols")
	if err != nil {
		log.Fatalln(miner.MinerID, "-", err)
	}
	defer kernelSolutions.Release()

	//Create memory buffers
	dbg := make([]byte, 8, 8)
	bufferDbg, err := context.CreateBufferUnsafe(cl.MemReadWrite|cl.MemCopyHostPtr, 8, unsafe.Pointer(&dbg[0]))
	if err != nil {
		log.Panicln(err)
	}
	defer bufferDbg.Release()

	var bufferHt [2]*cl.MemObject
	bufferHt[0] = mining.CreateEmptyBuffer(context, cl.MemReadWrite, htSize)
	defer bufferHt[0].Release()
	bufferHt[1] = mining.CreateEmptyBuffer(context, cl.MemReadWrite, htSize)
	defer bufferHt[1].Release()

	bufferSolutions := mining.CreateEmptyBuffer(context, cl.MemReadWrite, int(unsafe.Sizeof(solst{})))
	defer bufferSolutions.Release()

	var bufferRowCounters [2]*cl.MemObject
	bufferRowCounters[0] = mining.CreateEmptyBuffer(context, cl.MemReadWrite, numberOfRows)
	defer bufferRowCounters[0].Release()
	bufferRowCounters[1] = mining.CreateEmptyBuffer(context, cl.MemReadWrite, numberOfRows)
	defer bufferRowCounters[1].Release()

	for {
		start := time.Now()
		target, header, deprecationChannel, job, err := miner.Client.GetHeaderForWork()
		if err != nil {
			log.Println("ERROR fetching work -", err)
			time.Sleep(1000 * time.Millisecond)
			continue
		}
		continueMining := true
		if !continueMining {
			log.Println("Halting miner ", miner.MinerID)
			break
		}

		log.Println(target, header, deprecationChannel, job)

		// Process first BLAKE2b-400 block
		blake := &blake2b_state_t{}
		zcash_blake2b_init(blake, zcashHashLength, equihashParamN, equihashParamK)
		zcash_blake2b_update(blake, header[:128], false)
		bufferBlake, err := context.CreateBufferUnsafe(cl.MemReadOnly|cl.MemCopyHostPtr, 64, unsafe.Pointer(&blake.h[0]))
		if err != nil {
			log.Panicln(err)
		}
		var globalWorkgroupSize int
		var localWorkgroupSize int
		for round := 0; round < equihashParamK; round++ {
			// Now on every round!
			localWorkgroupSize = 256
			globalWorkgroupSize = numberOfRows / rowsPerUint
			kernelInitHt.SetArgBuffer(0, bufferHt[round%2])
			kernelInitHt.SetArgBuffer(1, bufferRowCounters[round%2])
			commandQueue.EnqueueNDRangeKernel(kernelInitHt, nil, []int{globalWorkgroupSize}, []int{localWorkgroupSize}, nil)

			if round == 0 {
				kernelRounds[round].SetArgBuffer(0, bufferBlake)
				kernelRounds[round].SetArgBuffer(1, bufferHt[round%2])
				kernelRounds[round].SetArgBuffer(2, bufferRowCounters[round%2])
				globalWorkgroupSize = selectWorkSizeBlake()
				kernelRounds[round].SetArgBuffer(3, bufferDbg)
			} else {
				kernelRounds[round].SetArgBuffer(0, bufferHt[(round-1)%2])
				kernelRounds[round].SetArgBuffer(1, bufferHt[round%2])
				kernelRounds[round].SetArgBuffer(2, bufferRowCounters[(round-1)%2])
				kernelRounds[round].SetArgBuffer(3, bufferRowCounters[round%2])
				globalWorkgroupSize = numberOfRows
				kernelRounds[round].SetArgBuffer(4, bufferDbg)
			}
			if round == equihashParamK-1 {
				kernelRounds[round].SetArgBuffer(5, bufferSolutions)
			}
			localWorkgroupSize = 64
			commandQueue.EnqueueNDRangeKernel(kernelRounds[round], nil, []int{globalWorkgroupSize}, []int{localWorkgroupSize}, nil)
		}
		kernelSolutions.SetArgBuffer(0, bufferHt[0])
		kernelSolutions.SetArgBuffer(1, bufferHt[1])
		kernelSolutions.SetArgBuffer(2, bufferSolutions)
		kernelSolutions.SetArgBuffer(3, bufferRowCounters[0])
		kernelSolutions.SetArgBuffer(4, bufferRowCounters[1])
		globalWorkgroupSize = numberOfRows
		commandQueue.EnqueueNDRangeKernel(kernelSolutions, nil, []int{globalWorkgroupSize}, []int{localWorkgroupSize}, nil)
		// read solutions
		solutionsFound := miner.verifySolutions(commandQueue, bufferSolutions, header, target, job)
		bufferBlake.Release()
		log.Println("Solutions found:", solutionsFound)

		hashRate := float64(solutionsFound) / (time.Since(start).Seconds() * 1000000)
		miner.HashRateReports <- &mining.HashRateReport{MinerID: miner.MinerID, HashRate: hashRate}
	}

}

func (miner *singleDeviceMiner) verifySolutions(commandQueue *cl.CommandQueue, bufferSolutions *cl.MemObject, header []byte, target []byte, job interface{}) (solutionsFound int) {

	sols := &solst{}

	// Most OpenCL implementations of clEnqueueReadBuffer in blocking mode are
	// good, except Nvidia implementing it as a wasteful busywait.
	commandQueue.EnqueueReadBuffer(bufferSolutions, true, 0, int(unsafe.Sizeof(*sols)), unsafe.Pointer(sols), nil)

	// let's check these solutions we just read...
	if sols.nr > maxSolutions {
		log.Printf("ERROR: %d (probably invalid) solutions were dropped!\n", sols.nr-maxSolutions)
		sols.nr = maxSolutions
	}
	for i := 0; uint32(i) < sols.nr; i++ {
		solutionsFound += miner.verifySolution(sols, i)
	}
	miner.submitSolution(sols, solutionsFound, header, target, job)

	return
}

func (miner *singleDeviceMiner) verifySolution(sols *solst, index int) int {
	inputs := sols.values[index]
	seenLength := (1 << (prefix + 1)) / 8
	seen := make([]uint8, seenLength, seenLength)
	var i uint32
	var tmp uint8
	// look for duplicate inputs
	for i = 0; i < (1 << equihashParamK); i++ {
		if inputs[i]/uint32(8) >= uint32(seenLength) {
			log.Printf("Invalid input retrieved from device: %d\n", inputs[i])
			sols.valid[index] = 0
			return 0
		}
		tmp = seen[inputs[i]/8]
		seen[inputs[i]/8] |= 1 << (inputs[i] & 7)
		if tmp == seen[inputs[i]/8] {
			// at least one input value is a duplicate
			sols.valid[index] = 0
			return 0
		}
	}
	// the valid flag is already set by the GPU, but set it again because
	// I plan to change the GPU code to not set it
	sols.valid[index] = 1
	// sort the pairs in place
	for level := 0; level < equihashParamK; level++ {
		for i := 0; i < (1 << equihashParamK); i += (2 << uint(level)) {
			len := 1 << uint(level)
			sortPair(inputs[i:i+len], inputs[i+len:i+(2*len)])
		}
	}
	return 1
}

func sortPair1(a, b []uint32) {
	needSorting := false
	var tmp uint32
	for i := 0; i < len(a); i++ {
		if needSorting || a[i] > b[i] {
			needSorting = true
			tmp = a[i]
			a[i] = b[i]
			b[i] = tmp
		} else {
			if a[i] < b[i] {
				return
			}
		}
	}
}

func (miner *singleDeviceMiner) submitSolution(solutions *solst, solutionsFound int, header []byte, target []byte, job interface{}) {
	for i := 0; i < int(solutions.nr); i++ {
		if solutions.valid[i] > 0 {
			log.Println("DEBUG: should submit solution:", solutions.values[i], header, target, job)
		}
	}

}
