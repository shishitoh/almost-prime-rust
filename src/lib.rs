//! 概素数(almost-prime)を計算する関数モジュール。
//!
//! # 概素数とは
//!
//! ある自然数nがk-概素数であるとは、
//! n = p_1 * p_2 * ... * p_n
//! を満たすp_1, ..., p_nが存在することを言う。
//! ただし、p_1, ..., p_nは互いに等しくても良い素数である。
//!
//! 例えば、k = 3に対してのk-概素数は小さい順から
//! 8 = 2 * 2 * 2, 12 = 2 * 2 * 3, 18 = 2 * 3 * 3,
//! 20 = 2 * 2 * 5, 27 = 3 * 3 * 3, 28 = 2 * 2 * 7, 30 = 2 * 3 * 5, ...
//! となる。

#![allow(unused)]

mod merge;
mod sieve;

/// 概素数を列挙し、Vec<usize>で返す。
///
/// * `k` - k-概素数のk。
/// * `i` - この値未満の概素数を列挙する。
///
/// # Examples
///
/// ```
/// use almost_prime::almprm;
///
/// assert_eq!(almprm(1, 20), vec![2, 3, 5, 7, 11, 13, 17, 19]);
/// assert_eq!(almprm(3, 30), vec![8, 12, 18, 20, 27, 28]);
/// ```
///
/// # Notes
///
/// この関数の実態はモジュール [`almprms`] で定義された関数
/// almprm1, almprm2, ...
/// のうちで最も高速な関数を返すだけのもの。
/// 現在の実装では [`almprms::almprm3`] を使用している。
#[inline]
pub fn almprm(k: usize, i: usize) -> Vec<usize> {
    almprms::almprm3(k, i)
}

pub mod almprms {
    // [`almprm`]に対してのリンクの貼り方がわからない

    //! 概素数列挙の実装をまとめたモジュール。
    //!
    //! ひとつ上の階層で`almprm`が定義されているので
    //! 基本的にはこのモジュールを使う必要はない。

    use crate::merge::{heap_merge, SizeOrdVec};
    use crate::sieve::sieve;
    use num_integer as integer;
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    use std::ops;

    /// i未満のすべての自然数を
    /// どのkに対するk-概素数かで振り分ける実装。
    pub fn almprm1(k: usize, i: usize) -> Vec<usize> {
        let mut ret: Vec<Vec<usize>> = vec![Vec::new(); k + 1];

        for n in 1..i {
            for v in ret.iter_mut() {
                if v.iter().any(|&a| n % a == 0) {
                    continue;
                }
                v.push(n);
                break;
            }
        }

        ret.swap_remove(k)
    }

    /// 各自然数iに対して、iの素因数の個数を求める実装。
    pub fn almprm2(k: usize, i: usize) -> Vec<usize> {
        if k == 0 {
            if i < 2 {
                return Vec::new();
            } else {
                return vec![1];
            }
        } else if k == 1 {
            // 無くても動きはするがこっちの方が圧倒的に速い
            return sieve(i).collect();
        } else {
            let mut pk: Vec<u8> = vec![0; i];

            for p in sieve(integer::div_ceil(i, 1 << (k - 1))) {
                let mut pi = p;
                let mut r = 1;
                while pi < i && r <= k + 1 {
                    for q in (pi..i).step_by(pi) {
                        pk[q] += 1;
                    }
                    pi *= p;
                    r += 1;
                }
            }
            pk.into_iter()
                .enumerate()
                .filter_map(|(idx, val)| if val as usize == k { Some(idx) } else { None })
                .collect()
        }
    }

    /// 始めに素数を列挙し、それらからk個を選び積を生成する実装。
    pub fn almprm3(k: usize, i: usize) -> Vec<usize> {
        if k == 0 {
            if i < 2 {
                return Vec::new();
            } else {
                return vec![1];
            }
        }

        let primes: Vec<usize> = sieve(integer::div_ceil(i, 1 << (k - 1))).collect();
        let mut pks: BinaryHeap<Reverse<SizeOrdVec<usize>>> = BinaryHeap::new();

        almprm3_impl(k, i, 1, &primes, &mut pks);

        heap_merge(pks).0
    }

    fn almprm3_impl(
        k: usize,
        i: usize,
        mul_p: usize,
        primes: &[usize],
        pks: &mut BinaryHeap<Reverse<SizeOrdVec<usize>>>,
    ) {
        // オーバーフロー、処理速度を考えなければ次の文と同値
        //
        // ```
        // let p_end = primes.partition_point(
        //     |&p| mul_p * p.powi(k) < i
        // );
        // ```
        let p_end = primes.partition_point(|&p| {
            p < (integer::div_ceil(i, mul_p) as f64)
                .powf(1.0 / (k as f64))
                .ceil() as usize
        });

        if k == 1 {
            // 本当はpksにはIteratorを持たせたい
            pks.push(Reverse(SizeOrdVec(
                primes[..p_end]
                    .iter()
                    .map(|&x| x * mul_p)
                    .collect::<Vec<usize>>(),
            )));
        } else {
            for p in 0..p_end {
                almprm3_impl(k - 1, i, mul_p * primes[p] as usize, &primes[p..], pks);
            }
        }
    }

    /// 始めに素数を列挙し、それらを基にk=2, 3, ...に対するk-概素数
    /// を順に作成する実装。
    pub fn almprm4(k: usize, i: usize) -> Vec<usize> {
        if k == 0 {
            if i < 2 {
                return Vec::new();
            } else {
                return vec![1];
            }
        } else if k == 1 {
            return sieve(i).collect();
        }

        let primes: Vec<usize> = sieve(integer::div_ceil(i, 1 << (k - 1))).collect();

        // 素数の積を生成する際に同じ数が重複して作成されるのを避けるために
        // 最小の素因数が2の集合、3の集合、... と別々に管理する。
        let mut p_prod: Vec<Vec<usize>> = Vec::new();

        for r in 2..=k {
            make_prod(r, k, i, &primes, &mut p_prod);
        }

        let mut heap = p_prod
            .into_iter()
            .map(|v| Reverse(SizeOrdVec(v)))
            .collect::<BinaryHeap<_>>();

        heap_merge(heap).0
    }

    fn make_prod(r: usize, k: usize, i: usize, primes: &[usize], p_prod: &mut Vec<Vec<usize>>) {
        if r == 2 {
            p_prod.clear();

            for (p_begin, &prime) in primes
                .binary_take_while(|p| {
                    // p * p * 2.pow((k-r) as u32) < i
                    p * p < integer::div_ceil(i, 1 << (k - r))
                })
                .iter()
                .enumerate()
            {
                p_prod.push(
                    primes[p_begin..]
                        .binary_take_while(|p| prime * p < integer::div_ceil(i, 1 << (k - r)))
                        .iter()
                        .map(|&p| prime * p)
                        .collect::<Vec<usize>>(),
                );
            }
        } else {
            let mut p_prod_impl: Vec<BinaryHeap<Reverse<SizeOrdVec<usize>>>> = Vec::new();

            for (p_begin, &p) in primes
                .binary_take_while(|p| {
                    // p.pow(r as u32) * 2.pow((k-r) as u32) < i
                    p.pow(r as u32) < integer::div_ceil(i, 1 << (k - r))
                })
                .iter()
                .enumerate()
            {
                p_prod_impl.push(BinaryHeap::new());

                for pp in p_prod[p_begin..]
                    .binary_take_while(|x| {
                        // p * x[0] * 2.pow((k-r) as u32) < i
                        x[0] < integer::div_ceil(integer::div_ceil(i, 1 << (k - r)), p)
                    })
                    .iter()
                {
                    p_prod_impl.last_mut().unwrap().push(Reverse(SizeOrdVec(
                        pp.binary_take_while(|x| {
                            // p * x * 2.pow((k-r) as u32) < i
                            *x < integer::div_ceil(integer::div_ceil(i, 1 << (k - r)), p)
                        })
                        .iter()
                        .map(|&x| x * p)
                        .collect(),
                    )));
                }
            }

            *p_prod = p_prod_impl
                .into_iter()
                .map(|heap| heap_merge(heap).0)
                .collect();
        }
    }

    trait BinaryTakeWhile: ops::Index<usize> + ops::Index<ops::RangeTo<usize>> {
        type Item;

        /// どんなi: usizeに対しても、もしfunc(self[i]) == falseであったら
        /// i <= j を満たす全てのj: usizeについてfunc(self[j]) == falseが成立する
        /// (前方が全てtrue, 後方が全てfalseと2分されているとき)とき、
        /// この関数はfunc(self[i]) == falseを満たす最小のiに対して&self[..i]を返す。
        /// もしself全体に渡ってfunc(self[i]) == trueであれば、そのときは&self[..self.len()]を返す。
        ///
        /// 事前条件が成り立たない場合、この関数は意味のある返り値を返さない。
        fn binary_take_while<F>(
            &self,
            func: F,
        ) -> &<Self as ops::Index<ops::RangeTo<usize>>>::Output
        where
            F: Fn(&Self::Item) -> bool;
    }

    impl<T> BinaryTakeWhile for [T] {
        type Item = T;

        #[inline]
        fn binary_take_while<F>(
            &self,
            func: F,
        ) -> &<Self as ops::Index<ops::RangeTo<usize>>>::Output
        where
            F: FnMut(&Self::Item) -> bool,
        {
            let p = self.partition_point(func);
            &self[..p]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::almprms;
    use crate::sieve::sieve;

    fn test_common<F>(f: F)
    where
        F: Fn(usize, usize) -> Vec<usize>,
    {
        assert_eq!(Vec::<usize>::new(), f(0, 0));
        assert_eq!(Vec::<usize>::new(), f(0, 1));
        assert_eq!(vec![1], f(0, 2));
        assert_eq!(vec![1], f(0, 1000));
        for i in 0..10 {
            assert_eq!(sieve(i), f(1, i));
        }
        assert_eq!(
            vec![4, 6, 9, 10, 14, 15, 21, 22, 25, 26, 33, 34, 35, 38, 39],
            f(2, 40)
        );
        assert_eq!(vec![16, 24, 36, 40, 54, 56], f(4, 60));
    }

    #[test]
    fn test_almprm1() {
        test_common(almprms::almprm1);
    }

    #[test]
    fn test_almprm2() {
        test_common(almprms::almprm2);
        assert_eq!(almprms::almprm1(3, 1000), almprms::almprm2(3, 1000));
        assert_eq!(almprms::almprm1(5, 1000), almprms::almprm2(5, 1000));
        assert_eq!(almprms::almprm1(10, 10000), almprms::almprm2(10, 10000));
    }

    #[test]
    fn test_almprm3() {
        test_common(almprms::almprm3);
        assert_eq!(almprms::almprm2(3, 1000), almprms::almprm3(3, 1000));
        assert_eq!(almprms::almprm2(5, 1000), almprms::almprm3(5, 1000));
        assert_eq!(almprms::almprm2(10, 10000), almprms::almprm3(10, 10000));
        assert_eq!(almprms::almprm2(16, 1000000), almprms::almprm3(16, 1000000));
    }

    #[test]
    fn test_almprm4() {
        test_common(almprms::almprm4);
        assert_eq!(almprms::almprm3(3, 1000), almprms::almprm4(3, 1000));
        assert_eq!(almprms::almprm3(5, 1000), almprms::almprm4(5, 1000));
        assert_eq!(almprms::almprm3(10, 10000), almprms::almprm4(10, 10000));
        assert_eq!(almprms::almprm3(16, 1000000), almprms::almprm4(16, 1000000));
    }
}
