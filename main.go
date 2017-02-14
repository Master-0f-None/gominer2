package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/kilo17/go-opencl/cl"
	"github.com/kilo17/gominer2/algorithms/sia"
	"github.com/kilo17/gominer2/algorithms/zcash"
	"github.com/kilo17/gominer2/mining"
)

//Version is the released version string of gominer
var Version = "0.6-Dev"

var intensity = 28
var devicesTypesForMining = cl.DeviceTypeGPU

func main() {
	log.SetOutput(os.Stdout)
	printVersion := flag.Bool("v", false, "Show version and exit")
	useCPU := flag.Bool("cpu", false, "If set, also use the CPU for mining, only GPU's are used by default")
	flag.IntVar(&intensity, "I", intensity, "Intensity")
	miningAlgorithm := flag.String("algo", "sia", "Mining algorithm, can be `sia` or `zcash`")
	host := flag.String("url", "localhost:9980", "daemon or server host and port, for stratum servers, use `stratum+tcp://<host>:<port>`")
	pooluser := flag.String("user", "payoutaddress.rigname", "username, most stratum servers take this in the form [payoutaddress].[rigname]")
	excludedGPUs := flag.String("E", "", "Exclude GPU's: comma separated list of devicenumbers")
	flag.Parse()

	if *printVersion {
		fmt.Println("gominer version", Version)
		os.Exit(0)
	}
	
	log.Println("Platform", deviceIds())
	
	//Filter the excluded devices
	miningDevices := make(map[int]*cl.Device)
	for i, device := range clDevices {
		if deviceExcludedForMining(i, *excludedGPUs) {
			continue
		}
		miningDevices[i] = device
	}

	nrOfMiningDevices := len(miningDevices)
	var hashRateReportsChannel = make(chan *mining.HashRateReport, nrOfMiningDevices*10)

	var miner mining.Miner
	if *miningAlgorithm == "zcash" {
		log.Println("Starting zcash mining")
		c := zcash.NewClient(*host, *pooluser)

		miner = &zcash.Miner{
			ClDevices:       miningDevices,
			HashRateReports: hashRateReportsChannel,
			Client:          c,
		}
	} else {
		log.Println("Starting SIA mining")
		c := sia.NewClient(*host, *pooluser)

		miner = &sia.Miner{
			ClDevices:       miningDevices,
			HashRateReports: hashRateReportsChannel,
			Intensity:       intensity,
			GlobalItemSize:  globalItemSize,
			Client:          c,
		}
	}
	miner.Mine()

	//Start printing out the hashrates of the different gpu's
	hashRateReports := make([]float64, nrOfMiningDevices)
	for {
		//No need to print at every hashreport, we have time
		for i := 0; i < nrOfMiningDevices; i++ {
			report := <-hashRateReportsChannel
			hashRateReports[report.MinerID] = report.HashRate
		}
		fmt.Print("\r")
		var totalHashRate float64
		for minerID, hashrate := range hashRateReports {
			fmt.Printf("%d-%.1f ", minerID, hashrate)
			totalHashRate += hashrate
		}
		fmt.Printf("Total: %.1f MH/s  ", totalHashRate)

	}
}
func GetDevices(platform *Platform, deviceType DeviceType) ([]*Device, error) {
	var deviceIds [maxDeviceCount]C.cl_device_id
	var numDevices C.cl_uint
	var platformId C.cl_platform_id
	if platform != nil {
		platformId = platform.id
	}
	if err := C.clGetDeviceIDs(platformId, C.cl_device_type(deviceType), C.cl_uint(maxDeviceCount), &deviceIds[0], &numDevices); err != C.CL_SUCCESS {
		return nil, toError(err)
	}
	if numDevices > maxDeviceCount {
		numDevices = maxDeviceCount
	}
	devices := make([]*Device, numDevices)
	for i := 0; i < int(numDevices); i++ {
		devices[i] = &Device{id: deviceIds[i]}
	}
	return devices, nil
}
