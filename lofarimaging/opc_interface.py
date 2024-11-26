from opcua import Client
import numpy as np

DEFAULT_URI = "http://lofar.eu"


def download_xst(subband: int, integration_time_s: int, url: str = 'localhost', port: int = 50000):
    """
    Download cross correlation statistics

    Args:
        subband (int): Subband number
        integration_time_s (int): Integration time in seconds
        url (str): URL to connect to, defaults to 'localhost'
        port (str): Port to connect to, defaults to 50000

    Returns:
        Tuple[datetime.datetime, np.ndarray, int]: UTC time, visibilities (shape n_ant x n_ant),
                                                   RCU mode

    Raises:
        RuntimeError: if in mixed RCU mode
    """
    client = Client("opc.tcp://{}:{}/".format(url, port), timeout=1000)
    client.connect()
    client.load_type_definitions()
    # objects = client.get_objects_node()
    idx = client.get_namespace_index(DEFAULT_URI)
    obj = client.get_root_node().get_child(["0:Objects",
                                            "{}:StationMetrics".format(idx),
                                            "{}:RCU".format(idx)])

    obstime, visibilities_opc, rcu_modes = obj.call_method("{}:record_cross".format(idx), subband, integration_time_s)

    client.close_session()
    client.close_secure_channel()

    rcu_modes_on = set([mode for mode in rcu_modes if mode != '0'])
    if len(rcu_modes_on) == 1:
        rcu_mode = int(rcu_modes_on.pop())
    elif len(rcu_modes_on) == 0:
        rcu_mode = 0
    else:
        raise RuntimeError("Multiple nonzero RCU modes are used, that's not supported yet")

    assert(len(visibilities_opc) == 2) # Real and complex part
    visibilities = np.array(visibilities_opc)[0] + 1j * np.array(visibilities_opc[1])

    return obstime, visibilities, rcu_mode
