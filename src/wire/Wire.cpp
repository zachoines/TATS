#include <fcntl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <string>

#include "Wire.h"


void Wire::begin(std::string device) {
    i2cDevice = open(device.c_str(), O_RDWR);
    if (i2cDevice == -1)
    {
        throw std::runtime_error("Could not open " + device);
    }
}

// Write to an I2C slave device's register:
void Wire::write8(uint8_t addr, uint8_t reg, uint8_t data) {
    int retval;
    uint8_t outbuf[2];
    struct i2c_rdwr_ioctl_data msgset[1];

    outbuf[0] = reg;
    outbuf[1] = data;

    struct i2c_msg messages[] = {
        { addr, 0, 2, outbuf }
    };

    msgset[0].msgs = messages;
    msgset[0].nmsgs = 1;

    if (ioctl(i2cDevice, I2C_RDWR, &msgset) < 0) {
        throw std::runtime_error("Could not open write to device");
    }
}

// Read the given I2C slave device's register and return the read value in `*result`:
void Wire::read8(uint8_t addr, uint8_t reg, uint8_t *result) {

    int retval;
    uint8_t outbuf[1], inbuf[1];
    struct i2c_rdwr_ioctl_data msgset[1];

    struct i2c_msg messages[] = {
        { addr, 0, 1, outbuf },
        { addr, I2C_M_RD | I2C_M_NOSTART, 1, inbuf },
    };

    msgset[0].msgs = messages;
    msgset[0].nmsgs = 2;

    outbuf[0] = reg;

    inbuf[0] = 0;

    *result = 0;
    if (ioctl(i2cDevice, I2C_RDWR, &msgset) < 0) {
        throw std::runtime_error("Could not read from device");
    }

    *result = inbuf[0];
}

void Wire::writeRead(int i2cAddress, unsigned char * pWriteData, int writeBytes, unsigned char * pReadData, int readBytes)
{
    struct i2c_msg msg[2]; // declare a two i2c_msg array
    struct i2c_rdwr_ioctl_data i2c_data; // declare our i2c_rdwr_ioctl_data structure
    int msgsUsed=0; // how many i2c_msg structures are used
    int result; // return value from ioctl() call
    
    // Assume the first i2c_msg is used for writing
    if(pWriteData && writeBytes) {
        msg[0].addr = i2cAddress;
        msg[0].flags = 0;
        msg[0].buf = pWriteData;
        msg[0].len = writeBytes;
        msgsUsed++;

        // Load the second i2c_msg for receiving if the first one is used
        if(pReadData && readBytes) {
            // Load up receive msg
            msg[1].addr = i2cAddress;
            msg[1].flags = I2C_M_RD;
            msg[1].buf = pReadData;
            msg[1].len = readBytes;
            msgsUsed++;
        }
    }
    else if(pReadData && readBytes) {
        // Load the first i2c_msg for receiving (no transmit data)
        msg[0].addr = i2cAddress;
        msg[0].flags = I2C_M_RD;
        msg[0].buf = pReadData;
        msg[0].len = readBytes;
        msgsUsed++;
    }
    // Continue if we have data to transfer
    if(msgsUsed) {
        // i2c_msg array is loaded up. Now load i2c_rdwr_ioctl_data structure.
        i2c_data.msgs = msg;
        i2c_data.nmsgs = msgsUsed;
    }
    
    // With our open file descriptor, perform I2C message transfers
    // This function call returns the number of i2c_msg structures processed,
    // or a negative error value
    if (ioctl(i2cDevice, I2C_RDWR, &i2c_data) < 0) {
        throw std::runtime_error("Could not read from device");
    }
}

int Wire::delay(int milli) {
	struct timespec ts;
    int res;

    if (milli < 0)
    {
        errno = EINVAL;
        return -1;
    }

    ts.tv_sec = milli / 1000;
    ts.tv_nsec = (milli % 1000) * 1000000;

    do {
        res = nanosleep(&ts, &ts);
    } while (res && errno == EINTR);

    return res;
    
    /*using namespace std;

    chrono::system_clock::time_point timePt =
        chrono::system_clock::now() + chrono::milliseconds(msec);

    this_thread::sleep_until(timePt); */
}


