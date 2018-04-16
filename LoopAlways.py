def TurnOnDevice():
    print('The device is running now!')
    device = 1
    return device


if __name__ == '__main__':
    flag1 = 0
    device_on = 0
    device_on = TurnOnDevice()
    print(device_on)

    print('Read the state of the device, '
          'if the device is runing, device_on = 1, '
          'else device_on = 0')

    # this is the outermost while loop, and it can help to keep the whole program running forever
    while flag1 != 1:
        # if the device is closed, then the program will be always waiting to turn on the device
        # if the device is running, then skip the following while loop and keep running the program of
        # image process
        while device_on != 1:
            print('Read the state of the device, '
                  'if the device is runing, device_on = 1, '
                  'else device_on = 0')

        # image processing part
        print('Here is the image processing part!')






