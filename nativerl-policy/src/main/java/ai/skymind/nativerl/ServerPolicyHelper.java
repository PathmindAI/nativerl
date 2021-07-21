package ai.skymind.nativerl;

import ai.skymind.nativerl.Exception.PathmindInvalidResponseException;
import ai.skymind.nativerl.util.ObjectMapperHolder;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;

import java.util.List;

import static java.net.HttpURLConnection.*;

public class ServerPolicyHelper implements PolicyHelper {
    private static class Action {
        private List<Integer> actions;
        private Double probability;
    }

    private ObjectMapper objectMapper = ObjectMapperHolder.getJsonMapper();
    private OkHttpClient client = null;

    @Override
    public float[] computeActions(float[] state) {
        throw new UnsupportedOperationException("Unsupported method for ServerPolicyHelper");
    }

    @Override
    public long[] computeDiscreteAction(float[] state) {
        throw new UnsupportedOperationException("Unsupported method for ServerPolicyHelper");
    }

    @Override
    public double[] computeActions(String baseUrl, String token, String postBody) {
        if (disablePolicyHelper) {
            return null;
        }

        try {
            if (client == null) {
                client = new OkHttpClient();
            }

            RequestBody requestBody = RequestBody.create(
                    MediaType.parse("application/json; charset=utf-8"), postBody);

            Request.Builder builder = new Request.Builder().url(buildPredictPath(baseUrl))
                    .addHeader("access-token", token)
                    .post(requestBody);
            Request request = builder.build();

            Response response = client.newCall(request).execute();
            if (response.isSuccessful()) {
                ResponseBody body = response.body();
                String bodyStr = body.string();
                if (body != null) {
                    int k = 0;
                    Action action = objectMapper.readValue(bodyStr, Action.class);
                    double[] actionArray = new double[action.actions.size()];
                    for (Integer a : action.actions) {
                        actionArray[k++] = (double)a;
                    }
                    return actionArray;
                }
            } else {
                switch (response.code()) {
                    case HTTP_UNAUTHORIZED:
                        throw new PathmindInvalidResponseException("Make sure your Policy Server is up and Policy Server URL is valid.");
                    case HTTP_FORBIDDEN:
                        throw new PathmindInvalidResponseException("Make sure your token is valid.");
                    case HTTP_NOT_FOUND:
                        throw new PathmindInvalidResponseException("You reached out to wrong path. Please contact Pathmind team.");
                    default:
                        throw new PathmindInvalidResponseException("Error Occurred " + response);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public static String buildPredictPath(String baseURL) {
        baseURL = baseURL.replaceAll("/$", "");

        return baseURL + "/predict/";
    }
}
