# reset
# echo "Run this file inside workspace!!"
# echo "Compiling..."
# g++ src/temp.cpp `pkg-config --cflags --libs opencv4`
# echo ""
echo "Running tests..."

echo "IMG 1"
./a.out test/img1.jpg
echo ""
echo "IMG 2"
./a.out test/img2.jpg
echo ""
echo "IMG 3"
./a.out test/img3.jpg
echo ""
echo "IMG 4"
./a.out test/img4.jpg
echo ""
echo "IMG 5"
./a.out test/img5.jpg
echo ""
echo "IMG 6"
./a.out test/img6.jpg
echo ""
echo "IMG 7"
#./a.out test/img7.jpg
echo ""
echo "IMG 8"
./a.out test/img8.jpg
echo ""