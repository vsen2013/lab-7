sh *Lab7_part1.sh

diff output1.txt  Lab7_part1_Q1_golden.txt > result1.txt

if [ -s result1.txt ]
then
	echo "Test1 Failed!"
else
	echo "Test1 Passed!"
fi

diff output2.txt  Lab7_part1_Q2_golden.txt > result2.txt

if [ -s result2.txt ]
then
	echo "Test2 Failed!"
else
	echo "Test2 Passed!"
fi

diff output3.txt  Lab7_part1_Q3_golden.txt > result3.txt

if [ -s result3.txt ]
then
	echo "Test3 Failed!"
else
	echo "Test3 Passed!"
fi


rm result1.txt
rm result2.txt
rm result3.txt


