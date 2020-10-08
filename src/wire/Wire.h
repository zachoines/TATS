
#include <map>

#ifndef _Wire
#define _Wire

class Wire {
private:

std::map<int, std::string> addressDevice;
std::map<std::string, int> deviceAddress;
int i2cDevice = -1;

public:
    Wire(std::string device = "/dev/i2c-1");
    void write8(uint8_t addr, uint8_t reg, uint8_t data);
    void read8(uint8_t addr, uint8_t reg, uint8_t *result);
    void writeRead(int i2cAddress, unsigned char * pWriteData, int writeBytes, unsigned char * pReadData, int readBytes);
    int delay(int mili);
};


#endif