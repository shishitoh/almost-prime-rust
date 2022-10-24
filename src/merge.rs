#![allow(unused)]

use std::collections::BinaryHeap;
use std::cmp::{ Ordering, Reverse };

mod iter_merge {
    pub struct MergedIter<T: Ord, I: Iterator<Item=T>, J: Iterator<Item=T>> {
        i: I,
        j: J,
        oi: Option<T>,
        oj: Option<T>,
    }

    impl<T, I, J> Iterator for MergedIter<T, I, J>
    where
        T: Ord,
        I: Iterator<Item=T>,
        J: Iterator<Item=T>,
    {
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            use std::cmp::Ordering;

            let oi: Option<T> = std::mem::replace(&mut self.oi, None);
            let oj: Option<T> = std::mem::replace(&mut self.oj, None);
            match (oi, oj) {
                (Some(si), Some(sj)) => {
                    match si.cmp(&sj) {
                        Ordering::Less | Ordering::Equal => {
                            let next = si;
                            self.oi = self.i.next();
                            self.oj = Some(sj);
                            Some(next)
                        }
                        Ordering::Greater => {
                            let next = sj;
                            self.oj = self.j.next();
                            self.oi = Some(si);
                            Some(next)
                        }
                    }
                },
                (Some(si), None) => {
                    let next = si;
                    self.oi = self.i.next();
                    Some(next)
                },
                (None, Some(sj)) => {
                    let next = sj;
                    self.oj = self.j.next();
                    Some(next)
                },
                (None, None) => None,
            }
        }
    }

    /// 2つのiteratorをmergeする
    ///
    /// # Examples
    ///
    /// assert_eq!(
    ///     iter_merge(
    ///         vec![1, 4, 5, 7, 12].into_iter(),
    ///         vec![2, 3, 5, 6, 9].into_iter()
    ///     ).collect::<Vec<i32>>(),
    ///     vec![1, 2, 3, 4, 5, 5, 6, 7, 9, 12]
    /// );
    pub fn iter_merge<T, I, J>(mut i: I, mut j: J) -> MergedIter<T, I, J>
    where
        T: Ord,
        I: Iterator<Item=T>,
        J: Iterator<Item=T>,
    {
        let oi = i.next();
        let oj = j.next();
        MergedIter{ i, j, oi, oj }
    }
}

/// Vecをlenによって比較するためのヘルパー構造体。
#[repr(transparent)]
pub struct SizeOrdVec<T: Ord> (pub Vec<T>);

impl<T: Ord> Ord for SizeOrdVec<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.len().cmp(&other.0.len())
    }
}

impl<T: Ord> PartialOrd for SizeOrdVec<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord> PartialEq for SizeOrdVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
    }
}

impl<T: Ord> Eq for SizeOrdVec<T> {}

/// 2つのソート済みのSizeOrdVecをmergeして新しいSizeOrdVecを作成する。
fn merge<T: Ord>(mut vec1: SizeOrdVec<T>, mut vec2: SizeOrdVec<T>)
    -> SizeOrdVec<T>
{
    let selfvec: Vec<T>
        = std::mem::replace(&mut vec1.0, Vec::default());
    let othervec: Vec<T>
        = std::mem::replace(&mut vec2.0, Vec::default());
    SizeOrdVec(
        iter_merge::iter_merge(
            selfvec.into_iter(),
            othervec.into_iter()
        ).collect::<Vec<T>>()
    )
}

/// 任意個のソート済みのSizeOrdVecをmergeする。
///
/// ReverseでSizeOrdVecを包んでいるのは高速化のため。
pub fn heap_merge<T: Ord>(mut heap: BinaryHeap<Reverse<SizeOrdVec<T>>>)
    -> SizeOrdVec<T>
{
    if heap.len() == 0 {
        return SizeOrdVec(Vec::new());
    }

    while heap.len() > 1 {
        let a = heap.pop().unwrap();
        let b = heap.pop().unwrap();
        // ここでa, bはheapで最も長さが短いVecなので
        // この関数中では、mergeにかかる時間は
        // 理論上最も短くなる。
        let c = Reverse(merge(a.0, b.0));
        heap.push(c);
    }
    heap.pop().unwrap().0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_heap_merge() {
        let mut h: BinaryHeap<Reverse<SizeOrdVec<i32>>> = BinaryHeap::new();
        h.push(Reverse(SizeOrdVec(vec![2, 3, 5])));
        h.push(Reverse(SizeOrdVec(vec![4, 12 ,25])));
        h.push(Reverse(SizeOrdVec(vec![1, 6 ,18])));
        assert_eq!(
            heap_merge(h).0,
            vec![1, 2, 3, 4, 5, 6, 12, 18, 25]
        );

        let mut h: BinaryHeap<Reverse<SizeOrdVec<i32>>> = BinaryHeap::new();
        h.push(Reverse(SizeOrdVec(vec![2, 3, 5])));
        h.push(Reverse(SizeOrdVec(vec![4, 12 ,25])));
        h.push(Reverse(SizeOrdVec(Vec::new())));
        assert_eq!(
            heap_merge(h).0,
            vec![2, 3, 4, 5, 12, 25]
        );

        let h: BinaryHeap<Reverse<SizeOrdVec<i32>>> = BinaryHeap::new();
        assert_eq!(
            heap_merge(h).0,
            Vec::new()
        );
    }
}
