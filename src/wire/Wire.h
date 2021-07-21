#include <fcntl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <mutex>

namespace control {
    #pragma once
    class Wire {
    private:
        int i2cDevice = -1; 
    public:
        Wire(std::string device = "/dev/i2c-1");
        void write8(uint8_t addr, uint8_t reg, uint8_t data);
        void read8(uint8_t addr, uint8_t reg, uint8_t *result);
        void writeRead(uint8_t addr, uint8_t* writeBuff, uint8_t numBytesToWrite, uint8_t* readBuff, uint8_t numBytesToRead);
        int delay(int mili);
    };
};