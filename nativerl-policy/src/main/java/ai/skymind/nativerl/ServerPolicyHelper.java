package ai.skymind.nativerl;

import ai.skymind.nativerl.util.ObjectMapperHolder;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;

import java.util.ArrayList;
import java.util.List;

public class ServerPolicyHelper implements PolicyHelper {
    private static class Action {
        private List<Integer> actions;
        private Double probability;
    }

    private ObjectMapper objectMapper = ObjectMapperHolder.getJsonMapper();

    @Override
    public float[] computeActions(float[] state) {
        throw new UnsupportedOperationException("Unsupported method for ServerPolicyHelper");
    }

    @Override
    public long[] computeDiscreteAction(float[] state) {
        throw new UnsupportedOperationException("Unsupported method for ServerPolicyHelper");
    }

    @Override
    public float[] computeActions(String url, String token, String postBody) {
        if (disablePolicyHelper) {
            return null;
        }

        try {
            OkHttpClient client = new OkHttpClient();

            RequestBody requestBody = RequestBody.create(
                    MediaType.parse("application/json; charset=utf-8"), postBody);

            Request.Builder builder = new Request.Builder().url(url)
                    .addHeader("access-token", token)
                    .post(requestBody);
            Request request = builder.build();

            Response response = client.newCall(request).execute();
            if (response.isSuccessful()) {
                ResponseBody body = response.body();
                String bodyStr = body.string();
                if (body != null) {
                    System.out.println("Response:" + bodyStr);
                    int k = 0;
                    Action action = objectMapper.readValue(bodyStr, Action.class);
                    float[] actionArray = new float[action.actions.size()];
                    for (Integer a : action.actions) {
                        actionArray[k++] = (float)a;
                    }
                    return actionArray;
                }
            } else {
                System.err.println("Error Occurred " + response);
                return null;
            }
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
        return null;
    }
}
