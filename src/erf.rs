const ERX: f32 = 8.45062911510467529297e-01; /* 0x3FEB0AC1, 0x60000000 */
/*
 * Coefficients for approximation to  erf on [0,0.84375]
 */
const EFX8: f32 = 1.02703333676410069053e+00; /* 0x3FF06EBA, 0x8214DB69 */
const PP0: f32 = 1.28379167095512558561e-01; /* 0x3FC06EBA, 0x8214DB68 */
const PP1: f32 = -3.25042107247001499370e-01; /* 0xBFD4CD7D, 0x691CB913 */
const PP2: f32 = -2.84817495755985104766e-02; /* 0xBF9D2A51, 0xDBD7194F */
const PP3: f32 = -5.77027029648944159157e-03; /* 0xBF77A291, 0x236668E4 */
const PP4: f32 = -2.37630166566501626084e-05; /* 0xBEF8EAD6, 0x120016AC */
const QQ1: f32 = 3.97917223959155352819e-01; /* 0x3FD97779, 0xCDDADC09 */
const QQ2: f32 = 6.50222499887672944485e-02; /* 0x3FB0A54C, 0x5536CEBA */
const QQ3: f32 = 5.08130628187576562776e-03; /* 0x3F74D022, 0xC4D36B0F */
const QQ4: f32 = 1.32494738004321644526e-04; /* 0x3F215DC9, 0x221C1A10 */
const QQ5: f32 = -3.96022827877536812320e-06; /* 0xBED09C43, 0x42A26120 */
/*
 * Coefficients for approximation to  erf  in [0.84375,1.25]
 */
const PA0: f32 = -2.36211856075265944077e-03; /* 0xBF6359B8, 0xBEF77538 */
const PA1: f32 = 4.14856118683748331666e-01; /* 0x3FDA8D00, 0xAD92B34D */
const PA2: f32 = -3.72207876035701323847e-01; /* 0xBFD7D240, 0xFBB8C3F1 */
const PA3: f32 = 3.18346619901161753674e-01; /* 0x3FD45FCA, 0x805120E4 */
const PA4: f32 = -1.10894694282396677476e-01; /* 0xBFBC6398, 0x3D3E28EC */
const PA5: f32 = 3.54783043256182359371e-02; /* 0x3FA22A36, 0x599795EB */
const PA6: f32 = -2.16637559486879084300e-03; /* 0xBF61BF38, 0x0A96073F */
const QA1: f32 = 1.06420880400844228286e-01; /* 0x3FBB3E66, 0x18EEE323 */
const QA2: f32 = 5.40397917702171048937e-01; /* 0x3FE14AF0, 0x92EB6F33 */
const QA3: f32 = 7.18286544141962662868e-02; /* 0x3FB2635C, 0xD99FE9A7 */
const QA4: f32 = 1.26171219808761642112e-01; /* 0x3FC02660, 0xE763351F */
const QA5: f32 = 1.36370839120290507362e-02; /* 0x3F8BEDC2, 0x6B51DD1C */
const QA6: f32 = 1.19844998467991074170e-02; /* 0x3F888B54, 0x5735151D */
/*
 * Coefficients for approximation to  erfc in [1.25,1/0.35]
 */
const RA0: f32 = -9.86494403484714822705e-03; /* 0xBF843412, 0x600D6435 */
const RA1: f32 = -6.93858572707181764372e-01; /* 0xBFE63416, 0xE4BA7360 */
const RA2: f32 = -1.05586262253232909814e+01; /* 0xC0251E04, 0x41B0E726 */
const RA3: f32 = -6.23753324503260060396e+01; /* 0xC04F300A, 0xE4CBA38D */
const RA4: f32 = -1.62396669462573470355e+02; /* 0xC0644CB1, 0x84282266 */
const RA5: f32 = -1.84605092906711035994e+02; /* 0xC067135C, 0xEBCCABB2 */
const RA6: f32 = -8.12874355063065934246e+01; /* 0xC0545265, 0x57E4D2F2 */
const RA7: f32 = -9.81432934416914548592e+00; /* 0xC023A0EF, 0xC69AC25C */
const SA1: f32 = 1.96512716674392571292e+01; /* 0x4033A6B9, 0xBD707687 */
const SA2: f32 = 1.37657754143519042600e+02; /* 0x4061350C, 0x526AE721 */
const SA3: f32 = 4.34565877475229228821e+02; /* 0x407B290D, 0xD58A1A71 */
const SA4: f32 = 6.45387271733267880336e+02; /* 0x40842B19, 0x21EC2868 */
const SA5: f32 = 4.29008140027567833386e+02; /* 0x407AD021, 0x57700314 */
const SA6: f32 = 1.08635005541779435134e+02; /* 0x405B28A3, 0xEE48AE2C */
const SA7: f32 = 6.57024977031928170135e+00; /* 0x401A47EF, 0x8E484A93 */
const SA8: f32 = -6.04244152148580987438e-02; /* 0xBFAEEFF2, 0xEE749A62 */
/*
 * Coefficients for approximation to  erfc in [1/.35,28]
 */
const RB0: f32 = -9.86494292470009928597e-03; /* 0xBF843412, 0x39E86F4A */
const RB1: f32 = -7.99283237680523006574e-01; /* 0xBFE993BA, 0x70C285DE */
const RB2: f32 = -1.77579549177547519889e+01; /* 0xC031C209, 0x555F995A */
const RB3: f32 = -1.60636384855821916062e+02; /* 0xC064145D, 0x43C5ED98 */
const RB4: f32 = -6.37566443368389627722e+02; /* 0xC083EC88, 0x1375F228 */
const RB5: f32 = -1.02509513161107724954e+03; /* 0xC0900461, 0x6A2E5992 */
const RB6: f32 = -4.83519191608651397019e+02; /* 0xC07E384E, 0x9BDC383F */
const SB1: f32 = 3.03380607434824582924e+01; /* 0x403E568B, 0x261D5190 */
const SB2: f32 = 3.25792512996573918826e+02; /* 0x40745CAE, 0x221B9F0A */
const SB3: f32 = 1.53672958608443695994e+03; /* 0x409802EB, 0x189D5118 */
const SB4: f32 = 3.19985821950859553908e+03; /* 0x40A8FFB7, 0x688C246A */
const SB5: f32 = 2.55305040643316442583e+03; /* 0x40A3F219, 0xCEDF3BE6 */
const SB6: f32 = 4.74528541206955367215e+02; /* 0x407DA874, 0xE79FE763 */
const SB7: f32 = -2.24409524465858183362e+01; /* 0xC03670E2, 0x42712D62 */

const SIGN_MASK: u32 = 2147483648;

#[inline]
pub fn fabs(x: f32) -> f32 {
    let abs_mask = !SIGN_MASK;
    f32::from_bits(x.to_bits() & abs_mask)
}

fn erfc1(x: f32) -> f32 {
    let s: f32;
    let p: f32;
    let q: f32;

    s = fabs(x) - 1.0;
    p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
    q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));

    1.0 - ERX - p / q
}

fn erfc2(ix: u32, mut x: f32) -> f32 {
    let s: f32;
    let r: f32;
    let big_s: f32;
    let z: f32;

    if ix < 0x3ff40000 {
        /* |x| < 1.25 */
        return erfc1(x);
    }

    x = fabs(x);
    s = 1.0 / (x * x);
    if ix < 0x4006db6d {
        /* |x| < 1/.35 ~ 2.85714 */
        r = RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))));
        big_s = 1.0
            + s * (SA1
                + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8)))))));
    } else {
        /* |x| > 1/.35 */
        r = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))));
        big_s =
            1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))));
    }
    z = f32::from_bits(0);

    f32::exp(-z * z - 0.5625) * f32::exp((z - x) * (z + x) + r / big_s) / x
}

/// Error function (f32)
///
/// Calculates an approximation to the “error function”, which estimates
/// the probability that an observation will fall within x standard
/// deviations of the mean (assuming a normal distribution).
pub fn erf(x: f32) -> f32 {
    let r: f32;
    let s: f32;
    let z: f32;
    let y: f32;
    let mut ix: u32;
    let sign: usize;

    ix = x.to_bits();
    sign = (ix >> 31) as usize;
    ix &= 0x7fffffff;
    if ix >= 0x7ff00000 {
        /* erf(nan)=nan, erf(+-inf)=+-1 */
        return 1.0 - 2.0 * (sign as f32) + 1.0 / x;
    }
    if ix < 0x3feb0000 {
        /* |x| < 0.84375 */
        if ix < 0x3e300000 {
            /* |x| < 2**-28 */
            /* avoid underflow */
            return 0.125 * (8.0 * x + EFX8 * x);
        }
        z = x * x;
        r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
        s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
        y = r / s;
        return x + x * y;
    }
    if ix < 0x40180000 {
        /* 0.84375 <= |x| < 6 */
        y = 1.0 - erfc2(ix, x);
    } else {
        let x1p_1022 = f32::from_bits(0x00800000);
        y = 1.0 - x1p_1022;
    }

    if sign != 0 { -y } else { y }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf() {
        // assert!(erf(f32::NAN).is_nan());
        assert_eq!(erf(f32::INFINITY), 1.0);
        assert_eq!(erf(f32::NEG_INFINITY), -1.0);
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(erf(-0.0), -0.0);
        assert_eq!(erf(2.0), 0.9953222650189527);
    }
}
