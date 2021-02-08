//! This module is a dummy module. It contains doctests that should fail to compile. It's used for
//! testing the DeriveCopy custom-derive macro and should not contain any actual code.
//!
//! ```compile_fail
//! #[macro_use]
//! extern crate rustacuda;
//! extern crate rustacuda_core;
//!
//! #[derive(Clone)]
//! struct ShouldFailTuple(Vec<u64>);
//! ```
//!
//! ```compile_fail
//! #[macro_use]
//! extern crate rustacuda;
//! extern crate rustacuda_core;
//!
//! #[derive(Clone)]
//! struct ShouldFailStruct{v: Vec<u64>}
//! ```
//!
//! ```compile_fail
//! #[macro_use]
//! extern crate rustacuda;
//! extern crate rustacuda_core;
//!
//! #[derive(Clone)]
//! enum ShouldFailTupleEnum {
//!     Unit,
//!     Tuple(Vec<u64>),
//! }
//! ```
//!
//! ```compile_fail
//! #[macro_use]
//! extern crate rustacuda;
//! extern crate rustacuda_core;
//!
//! #[derive(Clone)]
//! enum ShouldFailStructEnum {
//!     Unit,
//!     Struct{v: Vec<u64>},
//! }
//! ```
//!
//! ```compile_fail
//! #[macro_use]
//! extern crate rustacuda;
//! extern crate rustacuda_core;
//!
//! #[derive(Copy, Clone)]
//! union ShouldFailUnion {
//!     u: *const u64,
//!     o: *const i64,
//! }
//! ```
