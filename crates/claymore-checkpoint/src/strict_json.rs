//! Minimal strict JSON value used for strict checkpoint metadata parsing.
//!
//! `serde_json::Map` silently keeps the last value when a JSON object repeats
//! a key. Checkpoint validation must instead fail closed, so this parser keeps
//! object entries in order and rejects duplicate keys at every nesting level.

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;
use serde_json::Number;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(bytes).map_err(|error| error.to_string())
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Self::Object(entries) => Some(entries),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Self::Array(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    /// Read an unsigned integer without rounding. JSON exporters sometimes
    /// serialize byte counts as floating-point values with an integral value;
    /// accept those only while the value is exactly representable by f64.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Number(value) => value.as_u64().or_else(|| {
                let number = value.as_f64()?;
                const MAX_EXACT_F64_INTEGER: f64 = 9_007_199_254_740_991.0;
                (number.is_finite()
                    && number >= 0.0
                    && number.fract() == 0.0
                    && number <= MAX_EXACT_F64_INTEGER)
                    .then_some(number as u64)
            }),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::Bool(_) => "boolean",
            Self::Number(_) => "number",
            Self::String(_) => "string",
            Self::Array(_) => "array",
            Self::Object(_) => "object",
        }
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct StrictVisitor;

        impl<'de> Visitor<'de> for StrictVisitor {
            type Value = Json;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a strict JSON value")
            }

            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(Json::Null)
            }

            fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(Json::Null)
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(Json::Bool(value))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(Json::Number(Number::from(value)))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(Json::Number(Number::from(value)))
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                Number::from_f64(value)
                    .map(Json::Number)
                    .ok_or_else(|| E::custom("JSON number is not finite"))
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(Json::String(value.to_owned()))
            }

            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(Json::String(value))
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element()? {
                    values.push(value);
                }
                Ok(Json::Array(values))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut entries: Vec<(String, Json)> = Vec::new();
                while let Some(key) = map.next_key::<String>()? {
                    if entries.iter().any(|(existing, _)| existing == &key) {
                        return Err(de::Error::custom(format!(
                            "duplicate JSON object key {key:?}"
                        )));
                    }
                    let value = map.next_value()?;
                    entries.push((key, value));
                }
                Ok(Json::Object(entries))
            }
        }

        deserializer.deserialize_any(StrictVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_exact_integral_float_metadata_without_rounding() {
        assert_eq!(
            Json::parse(b"71903645408.0").unwrap().as_u64(),
            Some(71_903_645_408)
        );
        assert_eq!(Json::parse(b"1.5").unwrap().as_u64(), None);
        assert_eq!(Json::parse(b"9007199254740992.0").unwrap().as_u64(), None);
        assert_eq!(Json::parse(b"-1.0").unwrap().as_u64(), None);
    }

    #[test]
    fn rejects_duplicate_keys_at_every_level() {
        let error = Json::parse(br#"{"a":1,"a":2}"#).unwrap_err();
        assert!(error.contains("duplicate JSON object key"), "{error}");
        let error = Json::parse(br#"{"outer":{"x":1,"x":2}}"#).unwrap_err();
        assert!(error.contains("duplicate JSON object key"), "{error}");
    }

    #[test]
    fn preserves_order_and_types() {
        let value = Json::parse(br#"{"b":true,"a":[1,2.5,null,"x"]}"#).unwrap();
        let entries = value.as_object().unwrap();
        assert_eq!(entries[0].0, "b");
        assert_eq!(entries[1].0, "a");
        assert_eq!(entries[1].1.as_array().unwrap().len(), 4);
        assert!(!Json::parse(b"1e999").unwrap_err().is_empty());
    }
}
