package zcash

import (
	"fmt"
	"log"
	"time"
	"unsafe"
	"github.com/kilo17/go-opencl/cl"
	"github.com/kilo17/gominer2/clients"
	"github.com/kilo17/gominer2/mining"
//	"bytes"
//	"encoding/binary"

//	"encoding/hex"

)




// Miner actually mines :-)
type Miner struct {
	ClDevices       map[int]*cl.Device
	HashRateReports chan *mining.HashRateReport
	Client          clients.Client
}

//singleDeviceMiner actually mines on 1 opencl device
type singleDeviceMiner struct {
	ClDevice        *cl.Device
	MinerID         int
	HashRateReports chan *mining.HashRateReport
	Client          clients.Client
}

//Mine spawns a separate miner for each device defined in the CLDevices and feeds it with work
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

type Solst struct {
	nr             uint32
	likelyInvalids uint32
	valid          [maxSolutions]uint8
	Values         [maxSolutions][(1 << equihashParamK)]uint32
}
//todo: changed




func numberOfComputeUnits(gpu string) int {
	//if gpu == "rx480" {
	//}
	if gpu == "Fiji" {
		return 64
	}

	log.Panicln("Unknown GPU: ", gpu)
	return 0
}

func selectWorkSizeBlake() (workSize int) {
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

	bufferSolutions := mining.CreateEmptyBuffer(context, cl.MemReadWrite, int(unsafe.Sizeof(Solst{})))
	defer bufferSolutions.Release()

	var bufferRowCounters [2]*cl.MemObject
	bufferRowCounters[0] = mining.CreateEmptyBuffer(context, cl.MemReadWrite, numberOfRows)
	defer bufferRowCounters[0].Release()
	bufferRowCounters[1] = mining.CreateEmptyBuffer(context, cl.MemReadWrite, numberOfRows)
	defer bufferRowCounters[1].Release()

	for {
		start := time.Now()
		target, header, deprecationChannel, job, err := miner.Client.GetHeaderForWork()
	//	log.Println("///////////////////////GetHeaderForWork//////////////////////////////////////////////////")
	//	log.Println("target", target )
	//	log.Println("deprecationChannel", deprecationChannel )
	//	log.Println("header", header )
	//	log.Println("job", job )

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
	//	log.Println("header[:128]", header[:128] )

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

	Sols := &Solst{}

	// Most OpenCL implementations of clEnqueueReadBuffer in blocking mode are
	// good, except Nvidia implementing it as a wasteful busy work.
	commandQueue.EnqueueReadBuffer(bufferSolutions, true, 0, int(unsafe.Sizeof(*Sols)), unsafe.Pointer(Sols), nil)


	// let's check these solutions we just read...
	if Sols.nr > maxSolutions {
		log.Printf("ERROR: %d (probably invalid) solutions were dropped!\n", Sols.nr-maxSolutions)
		Sols.nr = maxSolutions
	}
	for i := 0; uint32(i) < Sols.nr; i++ {

		solutionsFound += miner.verifySolution(Sols, i)
	//	log.Println("solutionsFound", solutionsFound)
	//	log.Println("Sols", Sols)
	//	log.Println("iiiii", i)

	}
	miner.SubmitSolution(Sols, solutionsFound, header, target, job)

	return

}


func (miner *singleDeviceMiner) verifySolution(sols *Solst, index int) int {
	inputs := sols.Values[index]

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

func sortPair(a, b []uint32) {
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
//todo: changed
//todo 				miner.SubmitSolutionZEC(sols, solutionsFound, header, target, job)





func (miner *singleDeviceMiner) SubmitSolution(Solutions *Solst, solutionsFound int, header []byte, target []byte, job interface{}) {
	for i := 0; i < int(Solutions.nr); i++ {
		if Solutions.valid[i] > 0 {
			log.Println("Solutions", Solutions)

			log.Println("Solutions.Values", Solutions.Values)
			log.Println("solutions.values[i]", Solutions.Values[i])
			//todo  out		ZCASH_SOL_LEN-byte buffer where the solution will be stored= 1344 uint8_t
			//todo  inputs		array of 32-bit inputs
			//todo   n		number of elements in array = 512
			var inputs = Solutions.Values[i]
			var byte_pos uint32 = 0
			var bits_left uint32 = prefix + 1
			var x uint32 = 0

			var num int = 512
			var x_bits_used uint32 = 0
			slice := make([]uint32, 515)
			const MaxUint = ^uint32(0)
			for n := 0; n < 512; n++ {


			if bits_left >= 8-x_bits_used {
					x |= inputs[byte_pos] >> (bits_left - 8 + x_bits_used)
					bits_left -= 8 - x_bits_used
					x_bits_used = 8

				goto Label3
				}
			if bits_left > 0 {
				var mask uint32 = ^(MaxUint << (8 - x_bits_used))   // changed -1 to ^0
				mask = ((^mask) >> bits_left) & mask
				x |= (inputs[byte_pos] << (8 - x_bits_used - bits_left)) & mask
				x_bits_used += bits_left
				bits_left = 0
				goto Label2
				}
		Label2:
			 if bits_left <= 0 {
				 byte_pos++
				bits_left = prefix + 1

				 goto Label3
				}
		Label3:
			 if  x_bits_used == 8 {
				 slice = append(slice, x)
				 x = 0
				 x_bits_used = 0

				}

	}
			fmt.Printf("len=%d cap=%d slice=%v\n", len(slice), cap(slice), slice)
			fmt.Println("address of 0th element:", &slice[0])

			Solar(num, slice)

}
}
}
func Solar(num int, slice  []uint32){
	for i := 0; i < 2 ; i++ {
		var ttt = len(slice)
		var arr [1500]uint32
		copy(arr[:], slice[:ttt])
	log.Println("400", arr[400])
		log.Println("700", arr[700])
		log.Println("900", arr[900])
		log.Println("1100", arr[1100])
		log.Println("1500", arr[1450])
		d := fmt.Sprintf("%x", arr[700])
		fmt.Printf("Hex conf of '%d' is '%s'\n", i, d)
		fmt.Println(arr)

		//	fmt.Println(slice[i])

	//	h := fmt.Sprintf("%x", slice[i])
	//	fmt.Printf("Hex conf of '%d' is '%s'\n", i, h)





			}
		}







