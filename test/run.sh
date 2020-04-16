# reset
# echo "Run this file inside workspace!!"
# echo "Compiling..."
# echo "Util..."
# g++ src/faceAlignment/util.cpp -c `pkg-config --cflags --libs opencv4`
# echo "Face Alignment..."
# g++ src/faceAlignment/align.cpp -c `pkg-config --cflags --libs opencv4`
# echo "Main..."
# g++ src/main.cpp -c `pkg-config --cflags --libs opencv4`
# echo "Module1..."
# g++ src/faceDetection/module1.cpp -c `pkg-config --cflags --libs opencv4`
# echo "Executable"
# g++ util.o align.o module1.o main.o -o exe `pkg-config --cflags --libs opencv4`
# echo "Done"

reset
echo "Make sure you are running this inside build folder!"
echo "Starting Test 1..."
./exe ../test/img1.jpg
echo ""
echo "Starting Test 2..."
./exe ../test/img2.jpg
echo ""
echo "Starting Test 3..."
./exe ../test/img3.jpg
echo ""
echo "Starting Test 4..."
./exe ../test/img4.jpg
echo ""
echo "Starting Test 5..."
./exe ../test/img5.jpg
echo ""
echo "Starting Test 6..."
./exe ../test/img6.jpg
echo ""
echo "Starting Test 7..."
./exe ../test/img7.jpg
echo ""