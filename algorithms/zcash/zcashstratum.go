package zcash

import (
	"encoding/binary"
	"encoding/hex"
	"errors"
	"log"
	"strings"
	"sync"
	"github.com/kilo17/gominer2/clients"
	"github.com/kilo17/gominer2/clients/stratum"
//	"github.com/kilo17/GoEndian"

)

// zcash stratum as defined on https://github.com/str4d/zips/blob/23d74b0373c824dd51c7854c0e3ea22489ba1b76/drafts/str4d-stratum/draft1.rst

type stratumJob struct {
	JobID      string
	Version    []byte
	PrevHash   []byte
	MerkleRoot []byte
	Reserved   []byte
	Time       []byte
	Bits       []byte
	CleanJobs  bool

	ExtraNonce2 stratum.ExtraNonce2
}

//StratumClient is a zcash client using the stratum protocol
type StratumClient struct {
	connectionstring string
	User             string

	mutex           sync.Mutex // protects following
	stratumclient   *stratum.Client
	extranonce1     []byte
	extranonce2Size uint
	target          []byte
	currentJob      stratumJob
	clients.BaseClient
}
// NewClient creates a new StratumClient given a '[stratum+tcp://]host:port' connectionstring
func NewClient(connectionstring, pooluser string) (sc clients.Client) {
	if strings.HasPrefix(connectionstring, "stratum+tcp://") {
		connectionstring = strings.TrimPrefix(connectionstring, "stratum+tcp://")
	}
	  sc = &StratumClient{connectionstring: connectionstring, User: pooluser}

	return
}

//Start connects to the stratum server and processes the notifications
func (sc *StratumClient) Start() {
	sc.mutex.Lock()
	defer func() {
		sc.mutex.Unlock()
	}()

	sc.DeprecateOutstandingJobs()

	sc.stratumclient = &stratum.Client{}
	//In case of an error, drop the current stratumclient and restart
	sc.stratumclient.ErrorCallback = func(err error) {
		log.Println("Error in connection to stratumserver:", err)
		sc.stratumclient.Close()
		sc.Start()
	}

	sc.subscribeToStratumTargetChanges()
	sc.subscribeToStratumJobNotifications()

	//Connect to the stratum server
	log.Println("Connecting to", sc.connectionstring)
	sc.stratumclient.Dial(sc.connectionstring)

	//Subscribe for mining
	//Close the connection on an error will cause the client to generate an error, resulting in te errorhandler to be triggered
	result, err := sc.stratumclient.Call("mining.subscribe", []string{"gominer"})
	if err != nil {
		log.Println("ERROR Error in response from stratum", err)
		sc.stratumclient.Close()
		return
	}
	reply, ok := result.([]interface{})
	if !ok || len(reply) < 2 {
		log.Println("ERROR Invalid response from stratum", result)
		sc.stratumclient.Close()
		return
	}

	log.Println("extranonce1-before Hex", sc.extranonce1)   //todo


	//Keep the extranonce1 and extranonce2_size from the reply
	if sc.extranonce1, err = stratum.HexStringToBytes(reply[1]); err != nil {
		log.Println("ERROR Invalid extranonce1 from startum")
		sc.stratumclient.Close()
		return
	}
	sc.extranonce2Size = uint(32 - numberOfZeroBytes - len(sc.extranonce1))
	if sc.extranonce2Size < 0 {
		log.Println("ERROR Incompatible server, nonce1 too long")
		sc.stratumclient.Close()
		return
	}
	log.Println("len(sc.extranonce1)len(sc.extranonce1)len(sc.extranonce1)", len(sc.extranonce1))
	log.Println("sc.extranonce2Sizesc.extranonce2Sizesc.extranonce2Size", sc.extranonce2Size)
	log.Println("numberOfZeroBytesnumberOfZeroBytesnumberOfZeroBytesnumberOfZeroBytes", numberOfZeroBytes)

	//Authorize the miner
	_, err = sc.stratumclient.Call("mining.authorize", []string{sc.User, ""})
	if err != nil {
		log.Println("Unable to authorize:", err)
		sc.stratumclient.Close()
		return
	}

}

func (sc *StratumClient) subscribeToStratumTargetChanges() {
	sc.stratumclient.SetNotificationHandler("mining.set_target", func(params []interface{}) {

		if params == nil || len(params) < 1 {
			log.Println("ERROR No target parameter supplied by stratum server")
			return
		}
		var err error
		sc.target, err = stratum.HexStringToBytes(params[0])
		if err != nil {
			log.Println("ERROR Invalid target supplied by stratum server:", params[0])
		}

		log.Println("Stratum server changed target to", params[0])
	})
}

func (sc *StratumClient) subscribeToStratumJobNotifications() {
	sc.stratumclient.SetNotificationHandler("mining.notify", func(params []interface{}) {
		log.Println("New job received from stratum server")
		if params == nil || len(params) < 8 {
			log.Println("ERROR Wrong number of parameters supplied by stratum server")
			return
		}
log.Println("params-1", params)
		sj := stratumJob{}

		sj.ExtraNonce2.Size = sc.extranonce2Size

		var ok bool
		var err error
		if sj.JobID, ok = params[0].(string); !ok {
			log.Println("ERROR Wrong job_id parameter supplied by stratum server")
			return
		}
		if sj.Version, err = stratum.HexStringToBytes(params[1]); err != nil {
			log.Println("ERROR Wrong version parameter supplied by stratum server:", params[1])
			return
		}
		v := binary.LittleEndian.Uint32(sj.Version)
		if v != 4 {
			log.Println("ERROR Wrong version supplied by stratum server:", sj.Version)
			return
		}
		if sj.PrevHash, err = stratum.HexStringToBytes(params[2]); err != nil {
			log.Println("ERROR Wrong prevhash parameter supplied by stratum server")
			return
		}
		if sj.MerkleRoot, err = stratum.HexStringToBytes(params[3]); err != nil {
			log.Println("ERROR Wrong merkleroot parameter supplied by stratum server")
			return
		}
	//	if sj.Reserved, err = stratum.HexStringToBytes(params[5]); err != nil {
	//		log.Println("ERROR Wrong reserved parameter supplied by stratum server")
	//		return
	//	}
		if sj.Time, err = stratum.HexStringToBytes(params[5]); err != nil {
			log.Println("ERROR Wrong time parameter supplied by stratum server")
			return
		}

		if sj.Bits, err = stratum.HexStringToBytes(params[6]); err != nil {
			log.Println("ERROR Wrong bits parameter supplied by stratum server")
			return
		}
		if sj.CleanJobs, ok = params[7].(bool); !ok {
			log.Println("ERROR Wrong clean_jobs parameter supplied by stratum server")
			return
		}
//		log.Println("params-2", params)					//todo
//		log.Println(" After Hex - sc.extranonce1", sc.extranonce1)	//todo

		sc.addNewStratumJob(sj)
	})

}

func (sc *StratumClient) addNewStratumJob(sj stratumJob) {
	sc.mutex.Lock()
	defer sc.mutex.Unlock()
	sc.currentJob = sj
	if sj.CleanJobs {
		sc.DeprecateOutstandingJobs()
	}
	sc.AddJobToDeprecate(sj.JobID)
}

//GetHeaderForWork fetches new work
func (sc *StratumClient) GetHeaderForWork() (target, header []byte, deprecationChannel chan bool, job interface{}, err error) {
	sc.mutex.Lock()
	defer sc.mutex.Unlock()
	job = sc.currentJob


	if sc.currentJob.JobID == "" {
		err = errors.New("No job received from stratum server yet")
		return
	}

	deprecationChannel = sc.GetDeprecationChannel(sc.currentJob.JobID)

	target = sc.target


	header = make([]byte, 0, 140)
	header = append(header, sc.currentJob.Version...)    // 4 bytes
//	fmt.Printf("Step 1 - len=%d cap=%d slice=%v\n", len(header), cap(header), header)

	header = append(header, sc.currentJob.PrevHash...)   // 32 bytes = 36
//	fmt.Printf("Step 2 - len=%d cap=%d slice=%v\n", len(header), cap(header), header)

	header = append(header, sc.currentJob.MerkleRoot...) // 32 bytes = 68
//	fmt.Printf("Step 3 - len=%d cap=%d slice=%v\n", len(header), cap(header), header)

//	var y []byte = 00000000000000000000000000000000
//	sc.currentJob.Reserved =

	sc.currentJob.Reserved = header[68:100]

	header = append(header, sc.currentJob.Reserved...)   // 32 bytes = 100   todo - removed and replaced to insert bytes

//	fmt.Printf("Step 4 - len=%d cap=%d slice=%v\n", len(header), cap(header), header)

	header = append(header, sc.currentJob.Time...)       // 4 bytes = 104
//	fmt.Printf("Step 5 - len=%d cap=%d slice=%v\n", len(header), cap(header), header)

	header = append(header, sc.currentJob.Bits...)       // 4 bytes = 108
//	fmt.Printf("Step 6 - len=%d cap=%d slice=%v\n", len(header), cap(header), header)
//	bits := hex.EncodeToString(sc.currentJob.Bits)
//	log.Println("sc.currentJob.Bits...", bits)

	//Add a 32 bytes nonce
	header = append(header, sc.extranonce1...)		// X bytes approx. 118
//	fmt.Printf("Step 7 - len=%d cap=%d slice=%v\n", len(header), cap(header), header)
//	extranonce11 := hex.EncodeToString(sc.extranonce1)
//	log.Println("extranonce11", extranonce11)




	header = append(header, sc.currentJob.ExtraNonce2.Bytes()...) 	// X bytes approx. 128
//	fmt.Printf("Step 8 - len=%d cap=%d slice=%v\n", len(header), cap(header), header)
//	ExtraNonce22 := hex.EncodeToString(sc.currentJob.ExtraNonce2.Bytes())
//	log.Println("ExtraNonce22", ExtraNonce22)

//	log.Println("END - sc.extranonce1 - #3", sc.extranonce1)				//todo


//	log.Println("End sc.currentJob.ExtraNonce2 - ",	sc.currentJob.ExtraNonce2)				//todo

	sc.currentJob.ExtraNonce2.Increment()
	// Append the 12 `0` bytes
	for i := 0; i < numberOfZeroBytes; i++ {
		header = append(header, 0)
//		log.Println("End sc.currentJob.ExtraNonce2 hhhhhhhh- ",	sc.currentJob.ExtraNonce2)				//todo


	}
//	fmt.Printf("Header 1#St9#1 - len=%d cap=%d slice=%v\n", len(header), cap(header), header) // X bytes approx. 140
	return

}



	//SubmitHeader reports a solved header
	//TODO: extract nonce and equihash solution from the header
	//TODO: nonce := hex.EncodeToString(header[32:40])
	//log.Println("/////////////////////////////////////////////////////////////////////////")

func (sc *StratumClient) SubmitSolution(final string, solutionsFound int, OrganizedHeader []byte, target []byte, job interface{}) (err error) {
	//	log.Println("equihashsolution", equihashsolution)

	sj, _ := job.(stratumJob)


	//todo``````````````````````````````````````````````````````````````````````````````````````````````````````````

//	var LenX1= len(sc.extranonce1)

	//todo``````````````````````````````````````````````````````````````````````````````````````````````````````````
/*
		var LenX2= len(sj.ExtraNonce2.Bytes())
		var LenLeft= 32 - (LenX2 + LenX1)
		slice1 := make([]byte, LenX2)
		copy(slice1, sj.ExtraNonce2.Bytes())
		slice2 := make([]byte, LenLeft)
	slice1 = append(slice1, slice2...)

*/


//	target7 := reverse(OrganizedHeader[113:140])
//	fmt.Printf("target7   %02x\n", target7)
	encodedExtraNonce2 := hex.EncodeToString(OrganizedHeader[113:140])

//	encodedExtraNonce2 := hex.EncodeToString(header[113:140])
	log.Println("encodedExtraNonce2", encodedExtraNonce2)


	equihashsolution := final

	nTime := hex.EncodeToString(sj.Time)
		log.Println("nTime", nTime)

		sc.mutex.Lock()
		c := sc.stratumclient
		sc.mutex.Unlock()
		stratumUser := sc.User
//		log.Println("stratumUser", []string{stratumUser, sj.JobID, nTime, encodedExtraNonce2, equihashsolution})

		_, err = c.Call("mining.submit", []string{stratumUser, sj.JobID, nTime, encodedExtraNonce2, equihashsolution})
		if err != nil {
			log.Println("FUCK FUCK FUCK FUCK", stratumUser, sj.JobID, nTime, encodedExtraNonce2, equihashsolution)
			return
		}
		return
	}



func reverse2(numbers []uint8) []uint8{
	newNumbers := make([]uint8, len(numbers))
	for i, j := 0, len(numbers)-1; i < j; i, j = i+1, j-1 {
		newNumbers[i], newNumbers[j] = numbers[j], numbers[i]
	}
	return newNumbers
}


