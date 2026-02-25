import torch
from torch import nn
import math
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

# Import Mamba from mamba-ssm or fallback to local implementation
try:
    from mamba_ssm import Mamba
except ImportError:
    # Fallback to local implementation in tcfmamba.layers
    from tcfmamba.layers import Mamba


class TCFMamba(SequentialRecommender):
    """
    TCFMamba: Trajectory Collaborative Filtering Mamba for Debiased Point-of-Interest Recommendation

    Published at CIKM 2025 (Seoul, Republic of Korea).
    Paper: https://doi.org/10.1145/3746252.3761175

    This model integrates two key modules:
    1. JLSDR (Joint Learning of Static and Dynamic Representations):
       - Static: Encodes POI geographic coordinates via LAPE
       - Dynamic: Captures temporal patterns via PAPE and CF signals
       - Fusion: Unified representation learning for debiased recommendation

    2. PSMN (Preference State Mamba Network):
       - Mamba-based sequential encoder with linear-time complexity
       - Captures long-range dependencies in user trajectories
       - Residual connections + Feed-forward network

    Authors:
        Jin Qian, Shiyu Song, Xin Zhang, Dongjing Wang,
        He Weng, Haiping Zhang, Dongjin Yu

    Attributes:
        hidden_size (int): Embedding dimension
        num_layers (int): Number of PSMN layers
        d_state (int): Mamba SSM state dimension
        d_conv (int): Mamba convolution width
        expand (int): Mamba expansion factor
    """

    input_type = InputType.SEQ

    def __init__(self, config, dataset):
        super(TCFMamba, self).__init__(config, dataset)

        # Load configuration
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        # Mamba hyperparameters
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]

        # Time field configuration
        self.TIME_FIELD = config.get("TIME_FIELD", "timestamp")
        self.TIME_SEQ_FIELD = self.TIME_FIELD + config.get("LIST_SUFFIX", "_list")
        self.TIME_TERM = config.get("Term", "day")  # 'day', 'week', or 'month'

        # Location field configuration (configurable)
        self.LAT_FIELD = config.get("LATITUDE_FIELD", "latitude")
        self.LON_FIELD = config.get("LONGITUDE_FIELD", "longitude")

        # Setup location encoding
        self.locdim = self.hidden_size
        self.itemloc_embedding = self._init_location_embedding(dataset)

        # Base item embedding
        self.itembase_embedding = nn.Embedding(
            self.n_items, self.locdim, padding_idx=0
        )

        # Normalization and dropout
        self.Norm = nn.LayerNorm(self.locdim, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        # PSMN layers (Position-aware State Space Model Network)
        self.PSMN = nn.ModuleList([
            PSMN(
                d_model=self.locdim,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])

        # Loss function
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _init_location_embedding(self, dataset):
        """
        Initialize location-aware embeddings from dataset item features.

        This method:
        1. Extracts latitude and longitude from dataset.item_feat
        2. Applies spherical coordinate transformation
        3. Precomputes LAPE (Location-Aware Position Encoding)
        4. Creates frozen embedding layer
        """
        # Extract coordinates from dataset
        itemX, itemY = self._extract_coordinates(dataset)

        # Stack coordinates
        Locations = torch.stack([itemX, itemY], dim=1)

        # Compute LAPE
        ItemLocEnco = self.LAPE(Locations, self.locdim, self.device)

        # Create frozen embedding
        itemloc_embedding = nn.Embedding.from_pretrained(
            ItemLocEnco, freeze=True, padding_idx=0
        )

        return itemloc_embedding

    def _extract_coordinates(self, dataset):
        """
        Extract and transform latitude/longitude coordinates from dataset.

        Supports multiple field naming conventions:
        - latitude/longitude
        - lat/lon
        - Custom fields via config
        """
        # Try configured field names first
        lat_field = self.LAT_FIELD
        lon_field = self.LON_FIELD

        # Extract from item_feat
        if hasattr(dataset.item_feat, 'interactions'):
            # RecBole DataFrame format
            if lat_field in dataset.item_feat.columns and lon_field in dataset.item_feat.columns:
                latitude = torch.tensor(dataset.item_feat[lat_field].values, dtype=torch.float32)
                longitude = torch.tensor(dataset.item_feat[lon_field].values, dtype=torch.float32)
            else:
                # Try alternate field names
                alt_lat_fields = ['latitude', 'lat', 'Latitude', 'LAT']
                alt_lon_fields = ['longitude', 'lon', 'lng', 'Longitude', 'LON', 'LNG']

                lat_field_found = None
                lon_field_found = None

                for f in alt_lat_fields:
                    if f in dataset.item_feat.columns:
                        lat_field_found = f
                        break

                for f in alt_lon_fields:
                    if f in dataset.item_feat.columns:
                        lon_field_found = f
                        break

                if lat_field_found is None or lon_field_found is None:
                    raise ValueError(
                        f"Cannot find latitude/longitude fields in dataset. "
                        f"Available columns: {list(dataset.item_feat.columns)}. "
                        f"Please set LATITUDE_FIELD and LONGITUDE_FIELD in config."
                    )

                latitude = torch.tensor(dataset.item_feat[lat_field_found].values, dtype=torch.float32)
                longitude = torch.tensor(dataset.item_feat[lon_field_found].values, dtype=torch.float32)
        else:
            # Direct tensor access
            latitude = dataset.item_feat.get(lat_field, dataset.item_feat.get('latitude', dataset.item_feat.get('lat')))
            longitude = dataset.item_feat.get(lon_field, dataset.item_feat.get('longitude', dataset.item_feat.get('lon')))

        # Apply spherical transformation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        itemX, itemY = self._spherical_transform(latitude, longitude, device)

        # Store for reference
        self.register_buffer('itemX', itemX.clone())
        self.register_buffer('itemY', itemY.clone())

        return itemX, itemY

    def _spherical_transform(self, latitudes, longitudes, device, radius=6371):
        """
        Transform spherical coordinates (lat, lon) to Cartesian coordinates (x, y).

        Args:
            latitudes: Tensor of latitudes in degrees
            longitudes: Tensor of longitudes in degrees
            device: Computation device
            radius: Earth radius in km (default: 6371)

        Returns:
            x, y: Transformed coordinates in meters (relative to minimum)
        """
        lat_rad = latitudes.to(device) * (math.pi / 180)
        lon_rad = longitudes.to(device) * (math.pi / 180)

        # Convert to Cartesian coordinates (scaled to meters)
        x = radius * torch.cos(lat_rad) * torch.sin(lon_rad) * 1000
        y = radius * torch.cos(lat_rad) * torch.cos(lon_rad) * 1000

        # Normalize to positive coordinates (relative to minimum)
        min_x, min_y = x.min(), y.min()
        x -= min_x
        y -= min_y

        # Round and move to CPU
        x = torch.round(x).to(torch.int32).cpu()
        y = torch.round(y).to(torch.int32).cpu()

        torch.cuda.empty_cache()
        return x, y

    def LAPE(self, locations, locdim, device):
        """
        Location-Aware Position Encoding.

        Encodes geographic coordinates using sinusoidal encoding similar to Transformer
        positional encoding, but adapted for 2D spatial coordinates.

        Args:
            locations: Tensor of shape [N, 2] with (x, y) coordinates
            locdim: Embedding dimension
            device: Computation device

        Returns:
            Position encoding tensor of shape [N, locdim]
        """
        # Normalize coordinates (divide by 200 for ~1km precision)
        locations = (locations / 200.0).to(device)
        d = locdim // 2

        position_encoding = torch.zeros((locations.shape[0], locdim)).to(device)

        # Apply sinusoidal encoding for both x and y dimensions
        for i in range(0, d, 2):
            # X coordinate encoding
            position_encoding[:, i] = torch.sin(locations[:, 0] * (-i / d) ** 3)
            position_encoding[:, i + 1] = torch.cos(locations[:, 0] * (-(i + 1) / d) ** 3)

            # Y coordinate encoding
            position_encoding[:, d + i] = torch.sin(locations[:, 1] * (-i / d) ** 3)
            position_encoding[:, d + i + 1] = torch.cos(locations[:, 1] * (-(i + 1) / d) ** 3)

        return position_encoding.cpu()

    def PAPE(self, TimeTensor, TimeDim, Term='day'):
        """
        Period-Aware Position Encoding.

        Encodes temporal information using sinusoidal encoding based on
        periodicity (day/week/month).

        Args:
            TimeTensor: Tensor of timestamps [batch_size, seq_len, 1]
            TimeDim: Embedding dimension
            Term: Period type ('day', 'week', or 'month')

        Returns:
            Time encoding tensor of shape [batch_size, seq_len, TimeDim]
        """
        d = TimeDim
        term_seconds = {'day': 86400, 'week': 604800, 'month': 2592000}
        Threshold = term_seconds.get(Term, 86400)

        # Normalize time by period threshold
        TimeTensor = (TimeTensor.unsqueeze(2) % Threshold) / 4320  # Scale factor
        batch_size, seq_len = TimeTensor.shape[0], TimeTensor.shape[1]

        # Sinusoidal encoding
        i = torch.arange(0, d, device=self.device).float()
        sin_encoding = torch.sin(TimeTensor[:, :, 0:1] * (-i[::2] / d) ** 3)
        cos_encoding = torch.cos(TimeTensor[:, :, 0:1] * (-i[1::2] / d) ** 3)

        time_encoding = torch.zeros((batch_size, seq_len, d), device=self.device)
        time_encoding[:, :, 0::2] = sin_encoding
        time_encoding[:, :, 1::2] = cos_encoding

        return time_encoding

    def item_embedding(self, item_seq=None):
        """Get combined item embeddings (location + base)."""
        if item_seq is None:
            item_seq = torch.arange(self.n_items, device=self.device)

        return self.itemloc_embedding(item_seq) + self.itembase_embedding(item_seq)

    def forward(self, item_seq, item_seq_len, time_seq=None):
        """
        Forward pass through TCFMamba.

        Args:
            item_seq: Item sequence tensor [batch_size, seq_len]
            item_seq_len: Sequence lengths [batch_size]
            time_seq: Timestamp sequence [batch_size, seq_len] (optional)

        Returns:
            Sequence output tensor [batch_size, hidden_size]
        """
        # Location-based embeddings
        LocState = self.Norm(self.itemloc_embedding(item_seq))

        # Base item embeddings
        EmbState = self.Norm(self.itembase_embedding(item_seq))

        # Time-based embeddings (if available)
        if time_seq is not None:
            TimeState = self.Norm(self.PAPE(time_seq, self.locdim, self.TIME_TERM))
            ItemState = LocState + EmbState + TimeState
        else:
            ItemState = LocState + EmbState

        State = ItemState

        # Pass through PSMN layers
        for i in range(self.num_layers):
            State = self.PSMN[i](State)

        # Gather output at last position
        Seq_output = self.gather_indexes(State, item_seq_len - 1)
        return Seq_output

    def calculate_loss(self, interaction):
        """Calculate training loss."""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # Get time sequence if available
        time_seq = None
        if self.TIME_SEQ_FIELD in interaction:
            time_seq = interaction[self.TIME_SEQ_FIELD]

        item_output = self.forward(item_seq, item_seq_len, time_seq)

        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(item_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(item_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:  # CE loss
            test_item_emb = self.item_embedding()
            logits = torch.matmul(item_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        return loss

    def predict(self, interaction):
        """Predict scores for given items."""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        time_seq = None
        if self.TIME_SEQ_FIELD in interaction:
            time_seq = interaction[self.TIME_SEQ_FIELD]

        item_output = self.forward(item_seq, item_seq_len, time_seq)
        test_item_emb = self.item_embedding(test_item)

        scores = torch.mul(item_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """Predict scores for all items (full sort)."""
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        time_seq = None
        if self.TIME_SEQ_FIELD in interaction:
            time_seq = interaction[self.TIME_SEQ_FIELD]

        item_output = self.forward(item_seq, item_seq_len, time_seq)
        test_items_emb = self.item_embedding()

        scores = torch.matmul(item_output, test_items_emb.transpose(0, 1))
        return scores

    def gather_indexes(self, output, gather_index):
        """Gather output at specific indexes."""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output = output.gather(dim=1, index=gather_index)
        return output.squeeze(1)


class PSMN(nn.Module):
    """
    Position-aware State Space Model Network.

    A Mamba-based encoder layer with residual connections and feed-forward network.
    """

    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers

        # Mamba SSM layer
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = PFFN(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor):
        """Forward pass through PSMN layer."""
        # Handle different input dimensions
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

        # Mamba encoding
        hidden_states = self.mamba(input_tensor)

        # Residual connection with LayerNorm
        if self.num_layers == 1:
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)

        # Feed-forward network
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class PFFN(nn.Module):
    """
    Position-wise Feed-Forward Network.
    """

    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        """Forward pass through FFN."""
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
