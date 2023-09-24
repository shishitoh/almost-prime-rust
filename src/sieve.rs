#![allow(unused)]

pub use sieves::sieve3 as sieve;

pub mod sieves {
    /// 最もオーソドックスな実装。
    pub fn sieve1(i: usize) -> Box<dyn Iterator<Item = usize>> {
        if i < 3 {
            Box::new(std::iter::empty())
        } else {
            Box::new(
                {
                    let mut flags: Vec<bool> = vec![true; i];

                    flags[0] = false;
                    flags[1] = false;

                    let mut j: usize = 2;
                    for j in (2..).take_while(|&x| x * x < i) {
                        if flags[j] {
                            for s in (j * j..i).step_by(j) {
                                flags[s] = false;
                            }
                        }
                    }

                    flags
                }
                .into_iter()
                .enumerate()
                .filter_map(|(v, item)| if item { Some(v) } else { None }),
            )
        }
    }

    /// 2の倍数を飛ばす実装。
    pub fn sieve2(i: usize) -> Box<dyn Iterator<Item = usize>> {
        if i < 3 {
            Box::new(std::iter::empty())
        } else {
            let mut flags: Vec<bool> = vec![true; i / 2];

            flags[0] = false;

            // NOTE: ((i-1)+4-1)/4 は(i-1)を4で割って切り上げた値
            for j in (1..).take_while(|&x| x * x + x < ((i - 1) + 4 - 1) / 4) {
                if flags[j] {
                    for s in (2 * j * (j + 1)..i / 2).step_by(2 * j + 1) {
                        flags[s] = false;
                    }
                }
            }

            Box::new(
                [2].into_iter()
                    .chain(flags.into_iter().enumerate().filter_map(|(v, item)| {
                        if item {
                            Some(2 * v + 1)
                        } else {
                            None
                        }
                    })),
            )
        }
    }

    /// 2, 3, 5の倍数を飛ばす実装。
    ///
    /// 実装の詳細については
    /// [エラトステネスの篩の高速化](https://qiita.com/peria/items/a4ff4ddb3336f7b81d50)の
    /// (1), (2), (6)
    /// を参照。
    pub fn sieve3(i: usize) -> Box<dyn Iterator<Item = usize>> {
        use constants::{bitmasks, d_D, d_Dp_q, D_q, D};
        use countr_iter::CountrIter;

        if i < 3 {
            return Box::new(std::iter::empty());
        } else if i < 4 {
            return Box::new(std::iter::once(2));
        } else if i < 6 {
            return Box::new([2, 3].into_iter());
        }
        let size = (i - 1) / 30 + 1;
        let mut flags: Vec<u8> = vec![0xff; size];

        {
            let r: u8 = ((i - 1) % 30 + 1) as u8;
            *flags.last_mut().unwrap() = if r <= 1 {
                0b00000000
            } else if r <= 7 {
                0b00000001
            } else if r <= 11 {
                0b00000011
            } else if r <= 13 {
                0b00000111
            } else if r <= 17 {
                0b00001111
            } else if r <= 19 {
                0b00011111
            } else if r <= 23 {
                0b00111111
            } else if r <= 29 {
                0b01111111
            } else {
                0b11111111
            }
        }

        flags[0] &= 0b11111110;

        for m1 in 0..=((((i - 1) as f64).sqrt() / 30.0) as usize) {
            for idx_i1 in CountrIter(flags[m1]) {
                let mut idx_i2 = idx_i1;
                let mut m = 30 * m1 * m1
                    + 2 * m1 * D[idx_i1 as usize] as usize
                    + D_q[idx_i1 as usize] as usize;
                let bitmask: &[u8; 8] = &bitmasks[idx_i1 as usize];
                let g: [usize; 8] = [
                    m1 * d_D[0] as usize + d_Dp_q[idx_i1 as usize][0] as usize,
                    m1 * d_D[1] as usize + d_Dp_q[idx_i1 as usize][1] as usize,
                    m1 * d_D[2] as usize + d_Dp_q[idx_i1 as usize][2] as usize,
                    m1 * d_D[3] as usize + d_Dp_q[idx_i1 as usize][3] as usize,
                    m1 * d_D[4] as usize + d_Dp_q[idx_i1 as usize][4] as usize,
                    m1 * d_D[5] as usize + d_Dp_q[idx_i1 as usize][5] as usize,
                    m1 * d_D[6] as usize + d_Dp_q[idx_i1 as usize][6] as usize,
                    m1 * d_D[7] as usize + d_Dp_q[idx_i1 as usize][7] as usize,
                ];

                // ループアンローリング
                while m < size && idx_i2 != 0 {
                    flags[m] &= bitmask[idx_i2 as usize];
                    m += g[idx_i2 as usize];
                    idx_i2 = match idx_i2 {
                        7 => 0,
                        _ => idx_i2 + 1,
                    };
                }
                while m + 28 * m1 + (D[idx_i1 as usize] as usize) - 1 < size {
                    flags[m] &= bitmask[0];
                    m += g[0];
                    flags[m] &= bitmask[1];
                    m += g[1];
                    flags[m] &= bitmask[2];
                    m += g[2];
                    flags[m] &= bitmask[3];
                    m += g[3];
                    flags[m] &= bitmask[4];
                    m += g[4];
                    flags[m] &= bitmask[5];
                    m += g[5];
                    flags[m] &= bitmask[6];
                    m += g[6];
                    flags[m] &= bitmask[7];
                    m += g[7];
                }
                while m < size {
                    flags[m] &= bitmask[idx_i2 as usize];
                    m += g[idx_i2 as usize];
                    idx_i2 += 1;
                }
            }
        }

        Box::new(
            [2, 3, 5]
                .into_iter()
                .chain(flags.into_iter().enumerate().flat_map(|(i, flag)| {
                    CountrIter(flag).map(move |idx| 30 * i + D[idx as usize] as usize)
                })),
        )
    }

    mod constants {
        //! sieve3で使用する定数をまとめたモジュール。
        #![allow(non_upper_case_globals)]
        pub const D: [u8; 8] = [1, 7, 11, 13, 17, 19, 23, 29];

        // 説明用変数、使用しない
        // pub const n_D: [u8; 8] = [7, 11, 13, 17, 19, 23, 29, 31];

        // d_D[i] = n_D[i] - D[i];
        pub const d_D: [u8; 8] = [6, 4, 2, 4, 2, 4, 6, 2];

        // D_q[i] = (D[i] * D[i]) / 30;
        pub const D_q: [u8; 8] = [0, 1, 4, 5, 9, 12, 17, 28];

        // d_Dp_q[i][j] = (D[i] * n_D[j]) / 30 - (D[i] * D[j]) / 30;
        pub const d_Dp_q: [[u8; 8]; 8] = [
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [2, 2, 0, 2, 0, 2, 2, 1],
            [3, 1, 1, 2, 1, 1, 3, 1],
            [3, 3, 1, 2, 1, 3, 3, 1],
            [4, 2, 2, 2, 2, 2, 4, 1],
            [5, 3, 1, 4, 1, 3, 5, 1],
            [6, 4, 2, 4, 2, 4, 6, 1],
        ];

        // 説明用変数、使用しない
        /*
        fn i_D(i: usize) {
            match i {
                D[j] => j, // コンパイル通るかは知らない
                _  => panic!(""),
            }
        }
        */

        // 説明用変数, 使用しない
        /* Dp_r[i][j] = (D[i] * D[j]) % 30;
        pub const Dp_r: [[u8; 8]; 8] = [
            [1, 7, 11, 13, 17, 19, 23, 29],
            [7, 19, 17, 1, 29, 13, 11, 23],
            [11, 17, 1, 23, 7, 29, 13, 19],
            [13, 1, 23, 19, 11, 7, 29, 17],
            [17, 29, 7, 11, 19, 23, 1, 13],
            [19, 13, 29, 7, 23, 1, 17, 11],
            [23, 11, 13, 29, 1, 17, 19, 7],
            [29, 23, 19, 17, 13, 11, 7, 1],
        ];
        */

        // bitmasks[i][j] = 0xff - (1 << i_D(Dp_r[i][j]))
        pub const bitmasks: [[u8; 8]; 8] = [
            [0xfe, 0xfd, 0xfb, 0xf7, 0xef, 0xdf, 0xbf, 0x7f],
            [0xfd, 0xdf, 0xef, 0xfe, 0x7f, 0xf7, 0xfb, 0xbf],
            [0xfb, 0xef, 0xfe, 0xbf, 0xfd, 0x7f, 0xf7, 0xdf],
            [0xf7, 0xfe, 0xbf, 0xdf, 0xfb, 0xfd, 0x7f, 0xef],
            [0xef, 0x7f, 0xfd, 0xfb, 0xdf, 0xbf, 0xfe, 0xf7],
            [0xdf, 0xf7, 0x7f, 0xfd, 0xbf, 0xfe, 0xef, 0xfb],
            [0xbf, 0xfb, 0xf7, 0x7f, 0xfe, 0xef, 0xdf, 0xfd],
            [0x7f, 0xbf, 0xdf, 0xef, 0xf7, 0xfb, 0xfd, 0xfe],
        ];
    }

    pub mod countr_iter {

        // 実行されないドキュメンテーションコメント書きたいんだけど...

        /// u8に対して立っているビットの位置を小さい順に返すIterator。
        ///
        /// # Examples
        ///
        /// ```ignore
        /// use crate::sieve::sieves::countr_iter::CountrIter;
        /// assert_eq!(
        ///     CountrIter(0b01001101).collect::<Vec<i32>>(),
        ///     vec![0, 2, 3, 6]
        /// );
        /// ```
        #[repr(transparent)]
        pub struct CountrIter(pub u8);

        impl Iterator for CountrIter {
            type Item = u8;

            fn next(&mut self) -> Option<Self::Item> {
                const TABLE: [u8; 8] = [0, 1, 6, 2, 7, 5, 4, 3];
                const DE_BRUJIN_SEQ: u8 = 0b00011101;
                if self.0 == 0 {
                    None
                } else {
                    let data = self.0;
                    self.0 &= self.0 - 1;
                    let hash: usize = (((data & (!data + 1)).overflowing_mul(DE_BRUJIN_SEQ).0)
                        as u8
                        >> 5) as usize;
                    Some(TABLE[hash])
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::sieves;
    extern crate test;
    use test::Bencher;

    fn test_common<F, R>(f: F)
    where
        F: Fn(usize) -> R,
        R: Iterator<Item = usize>,
    {
        assert_eq!(Vec::<usize>::new(), f(0).collect::<Vec<_>>());
        assert_eq!(Vec::<usize>::new(), f(1).collect::<Vec<_>>());
        assert_eq!(Vec::<usize>::new(), f(2).collect::<Vec<_>>());
        assert_eq!(vec![2], f(3).collect::<Vec<_>>());
        assert_eq!(vec![2, 3], f(4).collect::<Vec<_>>());
        assert_eq!(vec![2, 3], f(5).collect::<Vec<_>>());
        assert_eq!(vec![2, 3, 5], f(6).collect::<Vec<_>>());
        assert_eq!(vec![2, 3, 5], f(7).collect::<Vec<_>>());
        assert_eq!(vec![2, 3, 5, 7], f(8).collect::<Vec<_>>());
        assert_eq!(vec![2, 3, 5, 7], f(9).collect::<Vec<_>>());
        assert_eq!(vec![2, 3, 5, 7], f(10).collect::<Vec<_>>());
    }

    // TODO: このテストは不完全
    fn iter_equals<T: std::fmt::Debug + std::cmp::Eq>(
        i: impl Iterator<Item = T>,
        j: impl Iterator<Item = T>,
    ) {
        for (a, b) in i.zip(j) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_sieve1() {
        test_common(sieves::sieve1);
        assert_eq!(
            vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29],
            sieves::sieve1(30).collect::<Vec<_>>()
        );
        assert_eq!(
            vec![
                2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
                83, 89, 97
            ],
            sieves::sieve1(100).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_sieve2() {
        test_common(sieves::sieve2);
        iter_equals(sieves::sieve1(100), sieves::sieve2(100));
        iter_equals(sieves::sieve1(1000), sieves::sieve2(1000));
        iter_equals(sieves::sieve1(1000000), sieves::sieve2(1000000));
    }

    #[test]
    fn test_sieve3() {
        test_common(sieves::sieve3);
        iter_equals(sieves::sieve2(100), sieves::sieve3(100));
        iter_equals(sieves::sieve2(1000), sieves::sieve3(1000));
        iter_equals(sieves::sieve2(10000000), sieves::sieve3(10000000));
    }

    fn bench_common<R: Iterator<Item = usize>>(f: impl Fn(usize) -> R) {
        f(1000).collect::<Vec<usize>>();
    }

    #[bench]
    fn bench_sieve1(b: &mut Bencher) {
        b.iter(|| bench_common(sieves::sieve1));
    }

    #[bench]
    fn bench_sieve2(b: &mut Bencher) {
        b.iter(|| bench_common(sieves::sieve2));
    }

    #[bench]
    fn bench_sieve3(b: &mut Bencher) {
        b.iter(|| bench_common(sieves::sieve3));
    }
}
