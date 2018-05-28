#include <cassert>
#include <iostream>
#include <limits>
#include <cmath>
#include "../include/farm.h"


void test(std::uint8_t* lhs_ptr, std::uint8_t* rhs_ptr,
        std::uint8_t* dst_ptr, std::size_t run_depth, std::size_t rows,
        std::int16_t* lhs_offset, std::int16_t* rhs_offset,
        std::int32_t res_offset, std::int32_t res_mul,
        std::int32_t res_shift)
{
    asm volatile(

    	// load lhs 1 x 8 block
    	"mov x0, %[lhs_ptr]\n"
    	"ld1 {v0.8b}, [x0]\n"
    	
        // load rhs 8 x 1 block
        "mov x1, %[rhs_ptr]\n"
        "ld1 {v26.8b}, [x1]\n"
        
        // clear accumulator registors
        "dup v8.4s, wzr\n"
        "ld1r {v24.8h}, [%[lhs_offset]]\n"
        "dup v9.4s, wzr\n"
        "ld1r {v25.8h}, [%[rhs_offset]]\n"
        "dup v3.4s, wzr\n"

        "uxtl v4.8h, v26.8b\n"
        "uxtl v0.8h, v0.8b\n"

        "add v4.8h, v4.8h, v25.8h\n"     
        "add v0.8h, v0.8h, v24.8h\n"

        //every element in int32
        "smlal  v8.4s, v0.4h, v4.4h\n"
        "smlal2 v9.4s, v0.8h, v4.8h\n"
        
        "dup v0.4s, %w[res_offset]\n"
        "dup v1.4s, %w[res_mul]\n"
        "dup v2.4s, %w[res_shift]\n"
#if 0
        //only v8.s[0] is valid, v8.s[1-3]=0 is invalid
        "addv s8, v8.4s\n"
        //apply shift
        "add v8.4s, v8.4s, v0.4s\n"
        //apply multiplier
        "mul v8.4s, v8.4s, v1.4s\n"
        "and v3.16b, v8.16b, v2.16b\n"
        //"sshr v3.4s, v3.4s, #31\n"
        //"sqadd v8.4s, v8.4s, v3.4s\n"
        //"srshl v8.4s, v8.4s, v2.4s\n"
        //word to half word, int32 -> int16        
        //"sqxtn v3.4h, v8.4s\n"
        //"mov v8.d[0], v3.d[0]\n"
        //"mov v8.d[1], v3.d[0]\n"
        //half word to bytes, int16 -> int8
        //"sqxtun v8.8b, v8.8h\n"
#endif
        "addv s8, v8.4s\n"
        //apply multiplier
        //"SQRDMULH Vd.4S,Vn.4S,Vm.4S"
        "sqrdmulh v8.4s, v8.4s, v1.4s\n"
        //"mul v8.4s, v8.4s, v1.4s\n"
        //apply shift
        "add v8.4s, v8.4s, v0.4s\n"

        
        //word to half word, int32 -> int16        
        //"sqxtn v3.4h, v8.4s\n"
        //half word to bytes, int16 -> int8
        //"sqxtun v8.8b, v8.8h\n"

//#if 0      
        "st1 {v3.s}[0], [%[dst_ptr]]\n"
        //"st1 {v2.s}[0], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"
        "st1 {v3.s}[1], [%[dst_ptr]]\n"
        //"st1 {v2.s}[1], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"
        "st1 {v3.s}[2], [%[dst_ptr]]\n"
        //"st1 {v2.s}[2], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"
        "st1 {v3.s}[3], [%[dst_ptr]]\n"
        //"st1 {v2.s}[3], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"
//#else
        "st1 {v8.s}[0], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"
        "st1 {v8.s}[1], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"
        "st1 {v8.s}[2], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"
        "st1 {v8.s}[3], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"
//#endif
        "st1 {v9.s}[0], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"
        "st1 {v9.s}[1], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"
        "st1 {v9.s}[2], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"
        "st1 {v9.s}[3], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"

        : // outputs
        [lhs_ptr] "+r"(lhs_ptr), 
        [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr),
        [run_depth] "+r"(run_depth)
        : // inputs
        [rows] "r"(rows), // this is used when storing the result
        [lhs_offset] "r"(lhs_offset),
        [rhs_offset] "r"(rhs_offset),
        [res_offset] "r"(res_offset),
        [res_mul] "r"(res_mul),
        [res_shift] "r"(res_shift)    
        : // clobbers
        "cc", "memory", "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29"
    );
}


#define TEST_ROW (1)
#define TEST_DEPTH (8)
#define TEST_COL (8)

int main()
{
    std::uint8_t lhs_data[TEST_ROW*TEST_DEPTH];
    std::uint8_t rhs_data[TEST_DEPTH*TEST_COL];
    std::uint8_t result_data[TEST_ROW*TEST_COL];
    std::uint8_t result_ref_data[TEST_ROW*TEST_COL];

    std::size_t run_depth = TEST_DEPTH;
    std::size_t rows = TEST_ROW;
    std::int16_t lhs_offset = 0;
    std::int16_t rhs_offset = 0;
    std::int32_t res_offset = 3;
    
#ifdef NEW_QUANTIZE_METHOD
    std::int32_t res_mul = static_cast<std::int32_t>(0.75*(1ll<<31));
#else
    std::int32_t res_mul = 10;
#endif    
    std::int32_t res_shift = 3;

    for(int i=0; i<TEST_ROW*TEST_DEPTH; i++)
    {
        lhs_data[i] = i;
    }
                
    for(int i=0; i<TEST_DEPTH*TEST_COL; i++)
    {
        rhs_data[i] = i;
    }

    for(int i=0; i<TEST_ROW*TEST_COL; i++)
    {
        result_data[i] = 0;
        result_ref_data[i] = 0;
    }

    farm::MatrixMap<farm::MapOrder::RowMajor> lhs(lhs_data, TEST_ROW, TEST_DEPTH);
    farm::MatrixMap<farm::MapOrder::ColMajor> rhs(rhs_data, TEST_DEPTH, TEST_COL);
    farm::MatrixMap<farm::MapOrder::ColMajor> result(result_data, TEST_ROW, TEST_COL);
    farm::MatrixMap<farm::MapOrder::ColMajor> result_ref(result_ref_data, TEST_ROW, TEST_COL);

    /*
    test(lhs_data, rhs_data, result_data, 
         run_depth, rows,
         &lhs_offset, &rhs_offset,
         res_offset, res_mul, -res_shift);
    */
    printf("lhs:\n");
    lhs.print();
    printf("rhs:\n");
    rhs.print();

    /*
    printf("gevv_kernel_run test\n");    
    farm::gevv_kernel_run(lhs_data, rhs_data, result_data, 
                    run_depth, 
                    (&lhs_offset), (&rhs_offset),
                    res_offset, res_mul, -res_shift);
    */
    printf("SingleThreadGemm test\n");
    farm::SingleThreadGemm(lhs, rhs, &result, 
                          lhs_offset, rhs_offset, res_offset,
                          res_mul, res_shift);

    printf("\nresult 8-bit:\n");
    result.print();
    //printf("\nresult 16-bit:\n");
    //result.print16();
    //printf("\nresult 32-bit:\n");
    //result.print32();

    // Calculate result_uint8 as the correct gemm output
    for (int i = 0; i < TEST_ROW; i++) {
        for (int j = 0; j < TEST_COL; j++) {
            int temp = 0;
            for (int t = 0; t < TEST_DEPTH; t++) {
                temp += (static_cast<int32_t>(lhs(i, t)) + lhs_offset) * 
                        (static_cast<int32_t>(rhs(t, j)) + rhs_offset);
            }
#ifdef NEW_QUANTIZE_METHOD   
            double mul = static_cast<double>(res_mul)/(1ll<<31);        
            double res = static_cast<double>((temp * mul) + res_offset) / 
                         static_cast<double>(pow(2.0, res_shift));
            printf("(%dx%d(%lf)+%d)>>%d = %lf\n", temp, res_mul, mul, res_offset, res_shift, res);             
#else
            double res = static_cast<double>((temp + res_offset) * res_mul) / 
                         static_cast<double>(pow(2.0, res_shift));
#endif
            if (res < 0.0) {
                result_ref(i, j) = 0;
            } else if (res > 255.0) {
                result_ref(i, j) = 255;
            } else {
                result_ref(i, j) = static_cast<std::uint8_t>(std::round(res));
            }
        }
    }
    printf("\nresult_ref 8-bit:\n");
    result_ref.print();


    return 1;
}

