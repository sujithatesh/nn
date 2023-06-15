#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <math.h>

int train[4][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {0, 0, 0}
};
#define train_count 4

// x1 x2
// x1*w1 + x2*w2 = y
// y - train[i][2] = d
// d*d / n = result
// result derivative
// w1 += learning rate == 0.0001 * result derivative;
// w2 += learning rate * result derivative;

double sigmoidf(float x){
    return 1.0f/ (1.0f + expf(-x));
}


float cost(float w1, float w2, float b) {
    float result = 0.0f;
    for (size_t i = 0; i < train_count; i++) {
        float x0 = train[i][0];
        float x1 = train[i][1];
        float y  = sigmoidf(x0 * w1 + x1 * w2 + b);
        float d = y - train[i][2];
        result += d * d;
    }
    result /= train_count;
    return result;
}

int main(void) {
    srand(733340);
    float w1 = (float)rand() / (float)RAND_MAX;
    float w2 = (float)rand() / (float)RAND_MAX;
    float w3 = (float)rand() / (float)RAND_MAX;
    float w4 = (float)rand() / (float)RAND_MAX;
    float w21 = (float)rand() / (float)RAND_MAX;
    float w22 = (float)rand() / (float)RAND_MAX;
    float b = (float)rand() / (float)RAND_MAX;

    float eps = 1e-3, rate = 1e-1;

    int i = 20000;
    while(i--){
        float c = cost(w1, w2, b);

        float dw1 = (cost(w1 + eps, w2, b) - c)/ eps;
        float dw2 = (cost(w1, w2 + eps, b) - c)/ eps;

        float dw3 = (cost(w3 + eps, w4, b) - c)/ eps;
        float dw4 = (cost(w3, w4 + eps, b) - c)/ eps;

        float dw21 = (cost(w21 + eps, w22, b) - c)/ eps;
        float dw22 = (cost(w21, w22 + eps, b) - c)/ eps;

        float db = (cost(w1, w2, b + eps) - c)/ eps;

        w1 -= rate*dw1;
        w2 -= rate*dw2;

        w3 -= rate*dw3;
        w4 -= rate*dw4;

        w21 -= rate*dw21;
        w22 -= rate*dw22;

        b -= rate*db;

        for(int j = 0; j< train_count; j++){
            float x1 = train[j][0];
            float x2 = train[j][1];

            float o1  = sigmoidf(x1 * w1 + x2 * w2 + b);
            float o2  = sigmoidf(x1 * w3 + x2 * w4 + b);

            float y = sigmoidf(o1 * w21 + o2 * w22 + b);
        }
        printf("%f\n",cost(w21, w22, b));
    }

    for(int i = 0; i < train_count; i++){
        float x0 = train[i][0];
        float x1 = train[i][1];
        float y  = sigmoidf(x0 * w21 + x1 * w22 + b);
        //printf("%d %d %f\n", train[i][0], train[i][1], y);
    }

    return 0;
}
