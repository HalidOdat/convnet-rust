use super::Net;

impl<'de> serde::Deserialize<'de> for Net {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        const FIELDS: &[&str] = &["layers", "final_layer"];
        enum Field {
            Layers,
            FinalLayer,
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
                        formatter.write_str("`layers` or `final_layer`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "layers" => Ok(Field::Layers),
                            "final_layer" => Ok(Field::FinalLayer),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct NetVisitor;

        impl<'de> serde::de::Visitor<'de> for NetVisitor {
            type Value = Net;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Net")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Net, V::Error>
            where
                V: serde::de::SeqAccess<'de>,
            {
                let layers = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let final_layer = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                Ok(Net::from_layers(layers, final_layer))
            }

            fn visit_map<V>(self, mut map: V) -> Result<Net, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut layers = None;
                let mut final_layer = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Layers => {
                            if layers.is_some() {
                                return Err(serde::de::Error::duplicate_field("layers"));
                            }
                            layers = Some(map.next_value()?);
                        }
                        Field::FinalLayer => {
                            if final_layer.is_some() {
                                return Err(serde::de::Error::duplicate_field("final_layer"));
                            }
                            final_layer = Some(map.next_value()?);
                        }
                    }
                }
                let layers = layers.ok_or_else(|| serde::de::Error::missing_field("layers"))?;
                let final_layer =
                    final_layer.ok_or_else(|| serde::de::Error::missing_field("final_layer"))?;
                Ok(Net::from_layers(layers, final_layer))
            }
        }

        deserializer.deserialize_struct("Net", FIELDS, NetVisitor)
    }
}
