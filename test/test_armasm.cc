#include <cassert>
#include <iostream>
#include <limits>
#include <cmath>
#include "../include/farm.h"
#include <arm_neon.h>


void print_uint8x8(uint8x8_t v, std::string comment) {
    printf("======== knife %s ========\n", comment.c_str());
    printf("%d ", vget_lane_u8(v, 0));
    printf("%d ", vget_lane_u8(v, 1));
    printf("%d ", vget_lane_u8(v, 2));
    printf("%d ", vget_lane_u8(v, 3));
    printf("%d ", vget_lane_u8(v, 4));
    printf("%d ", vget_lane_u8(v, 5));
    printf("%d ", vget_lane_u8(v, 6));
    printf("%d ", vget_lane_u8(v, 7));
    printf("\n");
}

void print_int16x4(int16x4_t v, std::string comment) {
    printf("======== knife %s ========\n", comment.c_str());
    printf("%d ", vget_lane_s16(v, 0));
    printf("%d ", vget_lane_s16(v, 1));
    printf("%d ", vget_lane_s16(v, 2));
    printf("%d ", vget_lane_s16(v, 3));
    printf("\n");
}


void print_int16x8(int16x8_t v, std::string comment) {
    printf("======== knife %s ========\n", comment.c_str());
    printf("%d ", vgetq_lane_s16(v, 0));
    printf("%d ", vgetq_lane_s16(v, 1));
    printf("%d ", vgetq_lane_s16(v, 2));
    printf("%d ", vgetq_lane_s16(v, 3));
    printf("%d ", vgetq_lane_s16(v, 4));
    printf("%d ", vgetq_lane_s16(v, 5));
    printf("%d ", vgetq_lane_s16(v, 6));
    printf("%d ", vgetq_lane_s16(v, 7));
    printf("\n");
}

void print_int32x4(int32x4_t v, std::string comment) {
    printf("======== knife %s ========\n", comment.c_str());
    printf("%d ", vgetq_lane_s32(v, 0));
    printf("%d ", vgetq_lane_s32(v, 1));
    printf("%d ", vgetq_lane_s32(v, 2));
    printf("%d ", vgetq_lane_s32(v, 3));
    printf("\n");
}



// int32_t MAC_I16X8(int16x8_t lhs_i16x8, int16x8_t rhs_i16x8, int32x4_t dst_i32x4){  
//     dst_i32x4 = vmlal_high_s16(dst_i32x4, lhs_i16x8, rhs_i16x8);
//     dst_i32x4 = vmlal_s16(dst_i32x4, vget_low_s16(lhs_i16x8), vget_low_s16(rhs_i16x8)); 
//     return vaddvq_s32(dst_i32x4);  
// }

#define MAC_I16X8(lhs_i16x8, rhs_i16x8, dst_i32x4){                                       \
    dst_i32x4 = vmlal_high_s16(dst_i32x4, lhs_i16x8, rhs_i16x8);                          \
    dst_i32x4 = vmlal_s16(dst_i32x4, vget_low_s16(lhs_i16x8), vget_low_s16(rhs_i16x8));   \
}




void test_intrinsics(std::uint8_t* lhs_ptr, std::uint8_t* rhs_ptr,
        std::uint8_t* dst_ptr, std::size_t run_depth, std::size_t rows,
        std::int16_t* lhs_offset, std::int16_t* rhs_offset,
        std::int32_t res_offset, std::int32_t res_mul,
        std::int32_t res_shift)
{

    assert(run_depth%8 == 0);
    // input lhs/rhs data with type uint8, so it safe to cast uint8 -> uint16 -> int16
    // int16x4_t vreinterpret_s16_u16 (uint16x4_t a)
    int32x4_t acc_i32x4_11 = vdupq_n_s32 (0);
    int32x4_t acc_i32x4_12 = vdupq_n_s32 (0);
    int32x4_t acc_i32x4_13 = vdupq_n_s32 (0);
    int32x4_t acc_i32x4_14 = vdupq_n_s32 (0);
    int32x4_t acc_i32x4_21 = vdupq_n_s32 (0);
    int32x4_t acc_i32x4_22 = vdupq_n_s32 (0);
    int32x4_t acc_i32x4_23 = vdupq_n_s32 (0);
    int32x4_t acc_i32x4_24 = vdupq_n_s32 (0);
    int32x4_t acc_i32x4_31 = vdupq_n_s32 (0);
    int32x4_t acc_i32x4_32 = vdupq_n_s32 (0);
    int32x4_t acc_i32x4_33 = vdupq_n_s32 (0);
    int32x4_t acc_i32x4_34 = vdupq_n_s32 (0);
    int32x4_t acc_i32x4_41 = vdupq_n_s32 (0);
    int32x4_t acc_i32x4_42 = vdupq_n_s32 (0);
    int32x4_t acc_i32x4_43 = vdupq_n_s32 (0);
    int32x4_t acc_i32x4_44 = vdupq_n_s32 (0);

    for(int c=0; c<run_depth; c+=8) {
        std::uint8_t* ptr_tmp = lhs_ptr + c;
        int16x8_t lhs_i16x8_1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr_tmp)));
        ptr_tmp += run_depth;
        int16x8_t lhs_i16x8_2 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr_tmp)));
        ptr_tmp += run_depth;
        int16x8_t lhs_i16x8_3 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr_tmp)));
        ptr_tmp += run_depth;
        int16x8_t lhs_i16x8_4 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr_tmp)));

        ptr_tmp = rhs_ptr;
        int16x8_t rhs_i16x8_1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr_tmp)));
        ptr_tmp += run_depth;
        int16x8_t rhs_i16x8_2 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr_tmp)));
        ptr_tmp += run_depth;
        int16x8_t rhs_i16x8_3 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr_tmp)));
        ptr_tmp += run_depth;
        int16x8_t rhs_i16x8_4 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ptr_tmp)));


        
        int32_t acc_i32_11,acc_i32_22,acc_i32_33,acc_i32_44;

        // apply offset

        int16x8_t lhs_offset_i16x8 = vld1q_dup_s16(lhs_offset);
        int16x8_t rhs_offset_i16x8 = vld1q_dup_s16(rhs_offset);
        lhs_i16x8_1 = vaddq_s16(lhs_i16x8_1, lhs_offset_i16x8);
        lhs_i16x8_2 = vaddq_s16(lhs_i16x8_2, lhs_offset_i16x8);
        lhs_i16x8_3 = vaddq_s16(lhs_i16x8_3, lhs_offset_i16x8);
        lhs_i16x8_4 = vaddq_s16(lhs_i16x8_4, lhs_offset_i16x8);
        rhs_i16x8_1 = vaddq_s16(rhs_i16x8_1, rhs_offset_i16x8);
        rhs_i16x8_2 = vaddq_s16(rhs_i16x8_2, rhs_offset_i16x8);
        rhs_i16x8_3 = vaddq_s16(rhs_i16x8_3, rhs_offset_i16x8);
        rhs_i16x8_4 = vaddq_s16(rhs_i16x8_4, rhs_offset_i16x8);

        
        // print_int16x8(lhs_i16x8_1, "lhs_i16x8_1");
        // print_int16x8(rhs_i16x8_1, "rhs_i16x8_1");


        MAC_I16X8(lhs_i16x8_1, rhs_i16x8_1, acc_i32x4_11);
        MAC_I16X8(lhs_i16x8_1, rhs_i16x8_2, acc_i32x4_12);
        MAC_I16X8(lhs_i16x8_1, rhs_i16x8_3, acc_i32x4_13);
        MAC_I16X8(lhs_i16x8_1, rhs_i16x8_4, acc_i32x4_14);
        MAC_I16X8(lhs_i16x8_2, rhs_i16x8_1, acc_i32x4_21);
        MAC_I16X8(lhs_i16x8_2, rhs_i16x8_2, acc_i32x4_22);
        MAC_I16X8(lhs_i16x8_2, rhs_i16x8_3, acc_i32x4_23);
        MAC_I16X8(lhs_i16x8_2, rhs_i16x8_4, acc_i32x4_24);
        MAC_I16X8(lhs_i16x8_3, rhs_i16x8_1, acc_i32x4_31);
        MAC_I16X8(lhs_i16x8_3, rhs_i16x8_2, acc_i32x4_32);
        MAC_I16X8(lhs_i16x8_3, rhs_i16x8_3, acc_i32x4_33);
        MAC_I16X8(lhs_i16x8_3, rhs_i16x8_4, acc_i32x4_34);
        MAC_I16X8(lhs_i16x8_4, rhs_i16x8_1, acc_i32x4_41);
        MAC_I16X8(lhs_i16x8_4, rhs_i16x8_2, acc_i32x4_42);
        MAC_I16X8(lhs_i16x8_4, rhs_i16x8_3, acc_i32x4_43);
        MAC_I16X8(lhs_i16x8_4, rhs_i16x8_4, acc_i32x4_44);
    }

    // row 1
    int16x8_t res_offset_i16x8 = vdupq_n_s16(res_offset);
    int32x4_t sum_row1 = vdupq_n_s32 (0);
    int32_t sum_i32_11 = vaddvq_s32(acc_i32x4_11);
    sum_row1 = vld1q_lane_s32(&sum_i32_11, sum_row1, 0);
    int32_t sum_i32_12 = vaddvq_s32(acc_i32x4_12);
    sum_row1 = vld1q_lane_s32(&sum_i32_12, sum_row1, 1);
    int32_t sum_i32_13 = vaddvq_s32(acc_i32x4_13);
    sum_row1 = vld1q_lane_s32(&sum_i32_13, sum_row1, 2);
    int32_t sum_i32_14 = vaddvq_s32(acc_i32x4_14);
    sum_row1 = vld1q_lane_s32(&sum_i32_14, sum_row1, 3);
    // print_int32x4(sum_row1, "sum_row1 add");
    // sum * res_mul / 2^31
    sum_row1 = vqrdmulhq_n_s32(sum_row1, res_mul);
    // print_int32x4(sum_row1, "sum_row1 vqrdmulhq_n_s32");
    sum_row1 = vshlq_n_s32(sum_row1, res_shift);
    // print_int32x4(sum_row1, "sum_row1 vshlq_n_s32");
    
    // row 2
    int32x4_t sum_row2 = vdupq_n_s32 (0);
    int32_t sum_i32_21 = vaddvq_s32(acc_i32x4_21);
    sum_row2 = vld1q_lane_s32(&sum_i32_21, sum_row2, 0);
    int32_t sum_i32_22 = vaddvq_s32(acc_i32x4_22);
    sum_row2 = vld1q_lane_s32(&sum_i32_22, sum_row2, 1);
    int32_t sum_i32_23 = vaddvq_s32(acc_i32x4_23);
    sum_row2 = vld1q_lane_s32(&sum_i32_23, sum_row2, 2);
    int32_t sum_i32_24 = vaddvq_s32(acc_i32x4_24);
    sum_row2 = vld1q_lane_s32(&sum_i32_24, sum_row2, 3);
    // print_int32x4(sum_row2, "sum_row2 add");
    sum_row2 = vqrdmulhq_n_s32(sum_row2, res_mul);
    sum_row2 = vshlq_n_s32(sum_row2, res_shift);
    // print_int32x4(sum_row2, "sum_row2 vshlq_n_s32");

    int16x8_t sum_row12_i16x8 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(sum_row1),sum_row2),res_offset_i16x8);
    // print_int16x8(sum_row12_i16x8, "sum_row12_i16x8");
    uint8x8_t sum_row12_u8x8 = vqmovun_s16(sum_row12_i16x8);
    // print_uint8x8(sum_row12_u8x8, "sum_row12_u8x8");
    vst1_lane_u8(dst_ptr, sum_row12_u8x8, 0);
    vst1_lane_u8(dst_ptr+1, sum_row12_u8x8, 1);
    vst1_lane_u8(dst_ptr+2, sum_row12_u8x8, 2);
    vst1_lane_u8(dst_ptr+3, sum_row12_u8x8, 3);
    dst_ptr += 8;
    vst1_lane_u8(dst_ptr, sum_row12_u8x8, 4);
    vst1_lane_u8(dst_ptr+1, sum_row12_u8x8, 5);
    vst1_lane_u8(dst_ptr+2, sum_row12_u8x8, 6);
    vst1_lane_u8(dst_ptr+3, sum_row12_u8x8, 7);


    // row 3
    int32x4_t sum_row3 = vdupq_n_s32 (0);
    int32_t sum_i32_31 = vaddvq_s32(acc_i32x4_31);
    sum_row2 = vld1q_lane_s32(&sum_i32_31, sum_row3, 0);
    int32_t sum_i32_32 = vaddvq_s32(acc_i32x4_32);
    sum_row2 = vld1q_lane_s32(&sum_i32_32, sum_row3, 1);
    int32_t sum_i32_33 = vaddvq_s32(acc_i32x4_33);
    sum_row2 = vld1q_lane_s32(&sum_i32_33, sum_row3, 2);
    int32_t sum_i32_34 = vaddvq_s32(acc_i32x4_34);
    sum_row3 = vld1q_lane_s32(&sum_i32_34, sum_row3, 3);
    sum_row3 = vqrdmulhq_n_s32(sum_row3, res_mul);
    sum_row3 = vshlq_n_s32(sum_row3, res_shift);
    
    // row 4
    int32x4_t sum_row4 = vdupq_n_s32 (0);
    int32_t sum_i32_41 = vaddvq_s32(acc_i32x4_41);
    sum_row4 = vld1q_lane_s32(&sum_i32_41, sum_row4, 0);
    int32_t sum_i32_42 = vaddvq_s32(acc_i32x4_42);
    sum_row4 = vld1q_lane_s32(&sum_i32_42, sum_row4, 1);
    int32_t sum_i32_43 = vaddvq_s32(acc_i32x4_43);
    sum_row4 = vld1q_lane_s32(&sum_i32_43, sum_row4, 2);
    int32_t sum_i32_44 = vaddvq_s32(acc_i32x4_44);
    sum_row4 = vld1q_lane_s32(&sum_i32_44, sum_row4, 3);
    sum_row4 = vqrdmulhq_n_s32(sum_row4, res_mul);
    sum_row4 = vshlq_n_s32(sum_row4, res_shift);

    int16x8_t sum_row34_i16x8 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(sum_row3),sum_row4),res_offset_i16x8);
    // print_int16x8(sum_row34_i16x8, "sum_row34_i16x8");
    uint8x8_t sum_row34_u8x8 = vqmovun_s16(sum_row34_i16x8);
    // print_uint8x8(sum_row34_u8x8, "sum_row34_u8x8");
    dst_ptr += 8;
    vst1_lane_u8(dst_ptr, sum_row34_u8x8, 0);
    vst1_lane_u8(dst_ptr+1, sum_row34_u8x8, 1);
    vst1_lane_u8(dst_ptr+2, sum_row34_u8x8, 2);
    vst1_lane_u8(dst_ptr+3, sum_row34_u8x8, 3);
    dst_ptr += 8;
    vst1_lane_u8(dst_ptr, sum_row34_u8x8, 4);
    vst1_lane_u8(dst_ptr+1, sum_row34_u8x8, 5);
    vst1_lane_u8(dst_ptr+2, sum_row34_u8x8, 6);
    vst1_lane_u8(dst_ptr+3, sum_row34_u8x8, 7);

}



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


#define TEST_ROW (4)
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
    std::int16_t lhs_offset = -1;
    std::int16_t rhs_offset = 2;
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

    
    // test(lhs_data, rhs_data, result_data, 
    test_intrinsics(lhs_data, rhs_data, result_ref_data, 
         run_depth, rows,
         &lhs_offset, &rhs_offset,
         res_offset, res_mul, -res_shift);
    printf("result_ref:\n");
    result_ref.print();

    farm::gemm_4_kernel_run(lhs.data(), rhs.data(),
                            result.data(), lhs.cols(), lhs.rows(),
                            &lhs_offset, &rhs_offset,
                            res_offset, res_mul,
                            -res_shift);
    
    printf("lhs_offset %d rhs_offset %d res_offset %d res_mul %d res_shift %d\n", 
            lhs_offset, rhs_offset, res_offset, res_mul, res_shift);
    printf("lhs:\n");
    lhs.print();
    printf("rhs:\n");
    rhs.print();
    printf("result:\n");
    result.print();

    /*
    printf("gevv_kernel_run test\n");    
    farm::gevv_kernel_run(lhs_data, rhs_data, result_data, 
                    run_depth, 
                    (&lhs_offset), (&rhs_offset),
                    res_offset, res_mul, -res_shift);
    */

#if 0
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
#endif
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
            double res = static_cast<double>(temp * mul)/static_cast<double>(pow(2.0, res_shift)) + res_offset;
            // printf("(%dx%d(%lf)+%d)>>%d = %lf\n", temp, res_mul, mul, res_offset, res_shift, res);             
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

