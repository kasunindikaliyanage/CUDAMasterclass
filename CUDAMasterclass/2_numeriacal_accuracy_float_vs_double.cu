#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

void float_accuracy_comparison()
{
	printf("float accuracy comparison \n");
	float a = 3.1415927f;
	float b = 3.1415928f;
	if (a == b)
	{
		printf("a is equal to b\n");
	}
	else
	{
		printf("a does not equal b\n");
	}
}

void double_accuracy_comparison()
{
	printf("\ndouble accuracy comparison \n");
	double a = 3.1415927;
	double b = 3.1415928;
	if (a == b)
	{
		printf("a is equal to b\n");
	}
	else
	{
		printf("a does not equal b\n");
	}
}

//int main()
//{
//	float_accuracy_comparison();
//	double_accuracy_comparison();
//
//	return 0;
//}