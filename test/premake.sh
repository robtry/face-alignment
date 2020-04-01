reset
echo "Run this file inside workspace!!"
echo "Compiling..."
echo "Util..."
g++ src/faceAlignment/util.cpp -c `pkg-config --cflags --libs opencv4`
echo "Face Alignment..."
g++ src/faceAlignment/align.cpp -c `pkg-config --cflags --libs opencv4`
echo "Main..."
g++ src/main.cpp -c `pkg-config --cflags --libs opencv4`
# echo "Module1..."
# g++ src/Module1.cpp -c `pkg-config --cflags --libs opencv4`
echo "Executable"
#g++ main.o Module1.o `pkg-config --cflags --libs opencv4`
g++ util.o align.o main.o `pkg-config --cflags --libs opencv4`
echo "Done"