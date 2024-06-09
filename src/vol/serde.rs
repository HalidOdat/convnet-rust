use super::Vol;

impl<'de> serde::Deserialize<'de> for Vol {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &["sx", "sy", "depth", "w"];
        enum Field {
            Sx,
            Sy,
            Depth,
            W,
        }

        impl<'de> serde::Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: serde::de::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> serde::de::Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        formatter.write_str("`sx`, `sy`, `depth` or `w`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "sx" => Ok(Field::Sx),
                            "sy" => Ok(Field::Sy),
                            "depth" => Ok(Field::Depth),
                            "w" => Ok(Field::W),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct VolVisitor;

        impl<'de> serde::de::Visitor<'de> for VolVisitor {
            type Value = Vol;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Vol")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Vol, V::Error>
            where
                V: serde::de::SeqAccess<'de>,
            {
                let sx = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let sy = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let depth = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                let w = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(3, &self))?;
                Ok(Vol {
                    sx,
                    sy,
                    depth,
                    w,
                    dw: vec![0.0; sx + sy + depth],
                })
            }

            fn visit_map<V>(self, mut map: V) -> Result<Vol, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut sx = None;
                let mut sy = None;
                let mut depth = None;
                let mut w = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Sx => {
                            if sx.is_some() {
                                return Err(serde::de::Error::duplicate_field("sx"));
                            }
                            sx = Some(map.next_value()?);
                        }
                        Field::Sy => {
                            if sy.is_some() {
                                return Err(serde::de::Error::duplicate_field("sy"));
                            }
                            sy = Some(map.next_value()?);
                        }
                        Field::Depth => {
                            if depth.is_some() {
                                return Err(serde::de::Error::duplicate_field("depth"));
                            }
                            depth = Some(map.next_value()?);
                        }
                        Field::W => {
                            if w.is_some() {
                                return Err(serde::de::Error::duplicate_field("w"));
                            }
                            w = Some(map.next_value()?);
                        }
                    }
                }
                let sx = sx.ok_or_else(|| serde::de::Error::missing_field("sx"))?;
                let sy = sy.ok_or_else(|| serde::de::Error::missing_field("sy"))?;
                let depth = depth.ok_or_else(|| serde::de::Error::missing_field("depth"))?;
                let w = w.ok_or_else(|| serde::de::Error::missing_field("w"))?;
                Ok(Vol {
                    sx,
                    sy,
                    depth,
                    w,
                    dw: vec![0.0; sx * sy * depth],
                })
            }
        }

        deserializer.deserialize_struct("Vol", FIELDS, VolVisitor)
    }
}
