package zcash

import (
	"fmt"
	"log"
	"time"
	"unsafe"
	"github.com/kilo17/go-opencl/cl"
	"github.com/kilo17/gominer2/clients"
	"github.com/kilo17/gominer2/mining"
	"github.com/kilo17/GoEndian"
	//	"os"
	"hash"
	"crypto/sha256"
	//	"os"
	"encoding/hex"
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
	Values         [maxSolutions][1 << equihashParamK]uint32
	Finalz		[512]uint32

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
		if err != nil {
			log.Println("ERROR fetching work -", err)
			//	msg = <-deprecationChannel
			time.Sleep(1000 * time.Millisecond)
			continue
		}
		var ggg = 1
		if ggg > 2 {
			log.Println("deprecationChannel", deprecationChannel)
		}
		blake := &blake2b_state_t{}


		zcash_blake2b_init(blake, zcashHashLength, equihashParamN, equihashParamK)
		zcash_blake2b_update_2(blake, false, header[:128])


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
		//	log.Println("Solutions found:", solutionsFound)

		hashRate := float64(solutionsFound) / (time.Since(start).Seconds() * 100000000)


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
		for i := 0; i < (1 << equihashParamK); i += 2 << uint(level) {

			len := 1 << uint(level)
			sortPair(inputs[i:i+len], inputs[i+len:i+(2*len)])


		}
	}
	for j := range inputs {
		sols.Finalz[j] = inputs[j]

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
var xxx uint32
var jj []byte
func (miner *singleDeviceMiner) SubmitSolution(Solutions *Solst, solutionsFound int, header []byte, target []byte, job interface{}) {
	// log.Println("solutions found ", solutionsFound )
	//	log.Println("header ",header )

	for i := 0; i < int(Solutions.nr); i++ {
		if Solutions.valid[i] > 0 {

			var inputs= Solutions.Finalz

			//		var inputs= [16]uint32{3257, 933560, 120482, 855408, 926622, 2063223, 1493414, 2060455, 128849, 1486455, 216187, 1287572, 308040, 1320625, 2022698, 2085613}

			//		var n uint32 = 8

			var n uint32 = 512

			var byte_pos uint32 = 0
			var bits_left uint32 = prefix + 1
			var x uint32 = 0
			var x_bits_used uint32 = 0
			slice := make([]uint32, 0)
			const MaxUint= ^uint32(0)

			for ; byte_pos < n; {

				if bits_left >= 8-x_bits_used {

					x |= inputs[byte_pos] >> (bits_left - 8 + x_bits_used)
					bits_left -= 8 - x_bits_used
					x_bits_used = 8

				} else if bits_left > 0 {
					var mask uint32 = ^(MaxUint << (8 - x_bits_used))
					mask = ((^mask) >> bits_left) & mask
					x |= (inputs[byte_pos] << (8 - x_bits_used - bits_left)) & mask
					x_bits_used += bits_left
					bits_left = 0
				} else if bits_left <= 0 {
					byte_pos++
					bits_left = prefix + 1

				}
				if x_bits_used == 8 {
					slice = append(slice, x)
					x = 0
					x_bits_used = 0
				}
			}

			//				log.Println("66666666 Store Encoded Solution DONE 666666666666666")
			sliceExtract := make([]byte, 0)

			for v := range slice {
				var Extract uint32 = slice[v]                        //todo
				Extract4th := make([]byte, 4)                        //todo
				endian.Endian.PutUint32(Extract4th, uint32(Extract)) //todo
				sliceExtract = append(sliceExtract, Extract4th[0])

			}

			Sha256_Header_Sol := make([]uint8, len(header))
			copy(Sha256_Header_Sol, header)
			var SolLength= []uint8("\xfd\x40\x05")
			var Solz= sliceExtract
			var Sha256_Submit []uint8

			Sha256_Submit = append(Sha256_Submit, Sha256_Header_Sol...)
			Sha256_Submit = append(Sha256_Submit, SolLength...)
			Sha256_Submit = append(Sha256_Submit, Solz...)

			FinalSol := make([]uint8, len(SolLength))
			copy(FinalSol, SolLength)
			FinalSol = append(FinalSol, Solz...)
			FinalSolHex := hex.EncodeToString(FinalSol)

			//	log.Println("target target target target", FinalSolHex)

			Sha256 := DoubleSHA(Sha256_Submit)

			Sha256Rev := reverse(Sha256)

			Target_Compare := cmp_target_256(target, Sha256Rev)
			//xxx = xxx + 1

			Sha256Header := DoubleSHA(header[113:140])
			Sha256JJ := DoubleSHA(jj)
			Header_Compare := cmp_target_256(Sha256Header, Sha256JJ)
		//	log.Println("Header_Compare", Header_Compare)

		//	log.Println("Sha256Rev", Sha256Rev)
		//	log.Println("target", target)

			if Target_Compare < 0 {
				//		log.Println("Hash is above target")
				return
			}
			if Target_Compare > 0 && Header_Compare != 0{
				log.Println("Hash is under target")
				log.Println("Target_Compare", Target_Compare)
				log.Println("Header_Compare", Header_Compare)
		//		log.Println("Sha256Rev", Sha256Rev)

		//		log.Println("Target_Compare", Target_Compare)
				go func() {
					copy(jj, header[113:140])
					log.Println("attempt attempt attempt attempt")
					if e := miner.Client.SubmitSolution(FinalSolHex, solutionsFound, header, target, job); e != nil {
						log.Println(miner.MinerID, "- Error submitting solution -", e)
					//		os.Exit(3)
					//	OldHeader = header[113:140]

					}
				}()
				return
			}

		}
	}
}
//	var FinalHeader = []uint8("\xfd\x40\x05")

func OrderHeader(header []byte)  []byte {

	var FinalHeader []uint8
	var given = header[0:113]
	var hash = header[113:128] // 15
	//hashReverse := reverse(hash)
	var zeros = header[128:140] // 12

	//	fmt.Printf("header %02x\n", header)
	//	fmt.Printf("Given %02x\n", given)
	//	fmt.Printf("zeros %02x\n", zeros)
	//	fmt.Printf("hash %02x\n", hash)
	//	fmt.Printf("hashReverse %02x\n", hashReverse)


	FinalHeader = append(FinalHeader, given...)
	FinalHeader = append(FinalHeader, zeros...)
	FinalHeader = append(FinalHeader, hash...)
	//	fmt.Printf("FinalHeader %02x\n", FinalHeader)


	return FinalHeader

}


func reverse(numbers []uint8) []uint8{
	newNumbers := make([]uint8, len(numbers))
	for i, j := 0, len(numbers)-1; i < j; i, j = i+1, j-1 {
		newNumbers[i], newNumbers[j] = numbers[j], numbers[i]
	}
	return newNumbers
}

func DoubleSHA(b []byte)([]byte){
	var h hash.Hash = sha256.New()
	h.Write(b)
	var h2 hash.Hash = sha256.New()
	h2.Write(h.Sum(nil))

	return h2.Sum(nil)
}


func cmp_target_256(a []uint8, b []uint8) int32 {
	//	log.Println("a", a)
	//	log.Println("b", b)

	//	for i := SHA256_TARGET_LEN - 1; i >= 0; i-- {
	for i := 0; i < 32; i++ {
		if a[i] != b[i] {
			var ddd = int32(a[i]) - int32(b[i])



			//	log.Println("ddd", ddd)

			return ddd
		}
	}
	return 0
}

func CompareSlices(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}

	if (a == nil) != (b == nil) {
		return false
	}

	for i, v := range a {
		if v != b[i] {
			return false
		}
	}

	return true
}