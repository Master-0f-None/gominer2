package main
//hello
import (
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/kilo17/go-opencl/cl"
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
	password := flag.String("pass", "x", "password, most stratum servers take this in the form [x]")

	excludedGPUs := flag.String("E", "", "Exclude GPU's: comma separated list of devicenumbers")
	flag.Parse()

	if *printVersion {
		fmt.Println("gominer version", Version)
		os.Exit(0)
	}

	if *useCPU {
		devicesTypesForMining = cl.DeviceTypeAll
	}

	platforms, err := cl.GetPlatforms()
	if err != nil {
		log.Panic(err)
	}

	clDevices := make([]*cl.Device, 0, 4)
	for _, platform := range platforms {
		log.Println("Platform", platform.Name())
		platormDevices, err := cl.GetDevices(platform, devicesTypesForMining)
		if err != nil {
			log.Println(err)
		}
		log.Println(len(platormDevices), "device(s) found:")
		for i, device := range platormDevices {
			log.Println(i, "-", device.Type(), "-", device.Name())
			clDevices = append(clDevices, device)
		}
	}

	if len(clDevices) == 0 {
		log.Println("No suitable opencl devices found")
		os.Exit(1)
	}

	//Filter the excluded devices
	miningDevices := make(map[int]*cl.Device)
	for i, device := range clDevices {
		if deviceExcludedForMining(i, *excludedGPUs) {
			continue
		}
		miningDevices[i] = device
	}

	nrOfMiningDevices := len(miningDevices)
	var hashRateReportsChannel = make(chan *mining.HashRateReport, nrOfMiningDevices*3)

	var miner mining.Miner
	if *miningAlgorithm == "zcash" {
		log.Println("Starting zcash mining")
		c := zcash.NewClient(*host, *pooluser, *password)

		miner = &zcash.Miner{
			ClDevices:       miningDevices,
			HashRateReports: hashRateReportsChannel,
			Client:          c,
		}
	} else {
		log.Println("Also NOT supported")
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

//deviceExcludedForMining checks if the device is in the exclusion list
func deviceExcludedForMining(deviceID int, excludedGPUs string) bool {
	excludedGPUList := strings.Split(excludedGPUs, ",")
	for _, excludedGPU := range excludedGPUList {
		if strconv.Itoa(deviceID) == excludedGPU {
			return true
		}
	}
	return false
}
